import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import PI, IntegratedAcquisitionFunction
from smac.initial_design import (
    AbstractInitialDesign,
    DefaultInitialDesign,
    LatinHypercubeInitialDesign,
    RandomInitialDesign,
    SobolInitialDesign,
)
from smac.model.gaussian_process import GaussianProcess, MCMCGaussianProcess
from smac.utils.configspace import convert_configurations_to_array
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from ..paths import is_safe_relative
from .base import (
    BaseOptimizer, OptimizationResult, OptimizerParam, TrialCollector,
    merge_stopping, rebase_history,
)
from ..splits import holdout
from .timing import STATUS_SUCCESS
from .trial import evaluate_trial

logging.getLogger("smac").setLevel(logging.WARNING)

# What to tell SMAC the budget is when the run has no trial cap — a deadline or
# a target score instead. It only sizes the initial design, and a run bounded by
# time still has to decide how much of itself to spend exploring; this is that
# guess. Not a setting: the share cap is the knob for the same idea, and two
# ways to say it would only disagree.
_UNBOUNDED_BUDGET = 100

#: The two search strategies, and the SMAC facade behind each. A facade is a
#: bundle — surrogate, acquisition function, maximizer, encoder — chosen to work
#: together, so this is one decision rather than four.
_STRATEGIES = {
    "gp": BlackBoxFacade,
    "rf": HyperparameterOptimizationFacade,
}

#: How the Gaussian process is fitted. `vanilla` maximizes the marginal
#: likelihood once; `mcmc` samples the kernel's own hyperparameters and carries
#: an ensemble of processes, which is better calibrated and an order of
#: magnitude slower — three walkers per kernel dimension over 250-step chains,
#: refitted on every ask.
_GP_MODELS = ("vanilla", "mcmc")

#: The settings that decide how many points are sampled before the model takes
#: over. Grouped because they are only legible together: two caps and a choice
#: of which one binds is three questions with one answer.
INITIAL_POINTS = "initial_points"


def _per_hyperparameter() -> int:
    """SMAC's own bound when no count is given: this many initial points per
    hyperparameter in the space. Read from its signature rather than copied, so
    the trial cap cannot silently stop matching what SMAC would have done."""
    return _signature_default(
        AbstractInitialDesign.__init__, "n_configs_per_hyperparameter") or 10

#: The designs whose points are a *sequence*, so that a bigger draw contains a
#: smaller one. Sobol is one by construction and `sample_configuration` draws
#: sequentially from a seeded generator, so a resume that asks for more points
#: gets the ones it already has plus the rest — SMAC skips what it has evaluated
#: and the totals come out the same as an uninterrupted run.
#:
#: Latin hypercube is not: it stratifies each dimension into `n` bins, so
#: changing `n` moves every point. Measured, none of a 3-point draw survives
#: into a 7-point one. A resume under it must therefore keep the size it had.
_EXTENDABLE_DESIGNS = frozenset({"sobol", "random", "default_only"})

_INITIAL_DESIGNS = {
    "sobol": SobolInitialDesign,
    "latin_hypercube": LatinHypercubeInitialDesign,
    "random": RandomInitialDesign,
    "default_only": DefaultInitialDesign,
}

# How many configurations to ask the surrogate about when answering "is anything
# left better than the incumbent?". The space is sampled rather than covered, so
# this trades a little cost for a tighter bound; a thousand predictions from a
# fitted GP is milliseconds beside a trial.
_CONFIDENCE_SAMPLES = 1000

# Trials before the surrogate is allowed to end a run. A GP fitted on a handful
# of points is confident in the way a straight line through two points is: the
# posterior is narrow because there is nothing to contradict it.
_CONFIDENCE_MIN_TRIALS = 10


#: Where each blank setting's default comes from: the facade method that would
#: have supplied the component, and the argument on it. Read out of the
#: signature at export time rather than copied, so this cannot drift from the
#: SMAC that is installed. The random-forest entries only resolve under `rf` —
#: `BlackBoxFacade.get_model` has no `n_trees` — and `_signature_default`
#: returns None for an argument that is not there, which is the truthful answer.
_FACADE_DEFAULTS = {
    "random_probability": ("get_random_design", "probability"),
    "challengers": ("get_acquisition_maximizer", "challengers"),
    "local_search_iterations": ("get_acquisition_maximizer", "local_search_iterations"),
    "retrain_after": ("get_config_selector", "retrain_after"),
    "rf_trees": ("get_model", "n_trees"),
    "rf_max_depth": ("get_model", "max_depth"),
    "rf_min_samples_split": ("get_model", "min_samples_split"),
    "rf_min_samples_leaf": ("get_model", "min_samples_leaf"),
    "rf_feature_ratio": ("get_model", "ratio_features"),
}


def _signature_default(function, argument):
    """*function*'s default for *argument*, or None if it has neither."""
    import inspect

    parameter = inspect.signature(function).parameters.get(argument)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _normal_cdf(z):
    """Standard normal CDF, without pulling scipy in for one function."""
    from math import erf, sqrt

    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(z)


class SMACOptimizer(BaseOptimizer):
    """Bayesian optimization via SMAC.

    Fits a model of the objective from the trials so far and picks the next
    configuration from it, rather than sampling blind. Which model, and how much
    of the budget goes on exploring before it takes over, are the settings that
    matter most; the rest are here for someone who already knows what they want.

    Every setting left unset is SMAC's own default *for the chosen strategy* —
    the two strategies disagree about several of them, and following whichever
    one is in use is better than imposing a number of ours on both.
    """

    name = "SMAC"

    #: What this optimizer was called when it was hardwired to the
    #: Gaussian-process facade. Every `.ihpo` written before the strategy
    #: became a setting records that name.
    aliases = ("SMAC (BlackBox)",)

    #: A fitted model of the objective is the whole point of this optimizer, so
    #: it can be asked how sure it is — see `incumbent_confidence`.
    supports_confidence_stopping = True

    #: And that model is state carried between runs, which a changed metric
    #: invalidates: it was fitted to costs from the other objective.
    fits_surrogate = True

    params_schema = [
        OptimizerParam("search_strategy", "Search strategy", "select", "gp",
                       choices=["gp", "rf"]),
        # How many points to sample before the model takes over. SMAC bounds
        # this two ways at once and takes the smaller: a share of the budget,
        # and a count. Both are exposed, either can be switched off, and the
        # smaller-of-the-two can be made the larger — see `_initial_points`.
        OptimizerParam("use_share_cap", "Use share cap", "bool", True,
                       group=INITIAL_POINTS),
        OptimizerParam("share_cap", "Share cap", "float", None,
                       min=0.01, max=1.0, group=INITIAL_POINTS,
                       enabled_by=("use_share_cap",)),
        OptimizerParam("use_trial_cap", "Use trial cap", "bool", True,
                       group=INITIAL_POINTS),
        OptimizerParam("trial_cap", "Trial cap", "int", None,
                       min=1, max=10_000, group=INITIAL_POINTS,
                       enabled_by=("use_trial_cap",)),
        OptimizerParam("initial_points_use_max", "Use the larger of the two",
                       "bool", False, group=INITIAL_POINTS,
                       enabled_by=("use_share_cap", "use_trial_cap")),
        OptimizerParam("random_probability", "Random trial rate",
                       "float", None, min=0.0, max=1.0),
        OptimizerParam("use_default_config", "Include the model's defaults",
                       "bool", False),

        OptimizerParam("initial_design", "Sampling method", "select",
                       "sobol", advanced=True,
                       choices=["sobol", "latin_hypercube", "random", "default_only"]),
        OptimizerParam("acquisition", "Acquisition function",
                       "select", "ei", advanced=True, choices=["ei", "pi"]),
        OptimizerParam("acquisition_xi", "Improvement margin", "float", 0.0,
                       min=0.0, max=1.0, advanced=True),
        OptimizerParam("challengers", "Candidates per trial", "int",
                       None, min=1, max=100_000, advanced=True),
        OptimizerParam("local_search_iterations", "Local search iterations", "int",
                       None, min=1, max=1_000, advanced=True),
        OptimizerParam("retrain_after", "Refit interval", "int",
                       None, min=1, max=100, advanced=True),

        # The surrogate itself, one set per strategy, each shown only under the
        # strategy it belongs to. Every name is prefixed and every label says
        # "surrogate" for a reason that is not tidiness: the demo Random Forest
        # *model* is tuned over `max_depth` and `min_samples_split`, and the
        # random forest *surrogate* has settings of the same names. Both can be
        # on screen at once, and confusing them would silently tune the wrong
        # thing.
        OptimizerParam("rf_trees", "Surrogate trees", "int", None,
                       min=2, max=1_000, advanced=True,
                       depends_on=("search_strategy", "rf")),
        OptimizerParam("rf_max_depth", "Surrogate tree depth", "int", None,
                       min=1, max=1_000, advanced=True,
                       depends_on=("search_strategy", "rf")),
        OptimizerParam("rf_min_samples_split", "Surrogate split threshold", "int",
                       None, min=2, max=100, advanced=True,
                       depends_on=("search_strategy", "rf")),
        OptimizerParam("rf_min_samples_leaf", "Surrogate leaf size", "int", None,
                       min=1, max=100, advanced=True,
                       depends_on=("search_strategy", "rf")),
        # Capped at 1.0, and not only for tidiness: above it SMAC computes
        # `max_features = 0` and the forest splits on nothing at all.
        OptimizerParam("rf_feature_ratio", "Surrogate feature ratio", "float",
                       None, min=0.05, max=1.0, advanced=True,
                       depends_on=("search_strategy", "rf")),
        OptimizerParam("rf_bootstrapping", "Bootstrap the surrogate's trees",
                       "bool", True, advanced=True,
                       depends_on=("search_strategy", "rf")),

        OptimizerParam("gp_model_type", "Surrogate fitting", "select", "vanilla",
                       choices=["vanilla", "mcmc"], advanced=True,
                       depends_on=("search_strategy", "gp")),
        OptimizerParam("gp_restarts", "Surrogate fit restarts", "int", None,
                       min=1, max=100, advanced=True,
                       depends_on=("search_strategy", "gp")),
        OptimizerParam("gp_normalize_y", "Normalise surrogate targets", "bool",
                       True, advanced=True,
                       depends_on=("search_strategy", "gp")),
    ]

    def __init__(self, search_strategy="gp",
                 use_share_cap=True, share_cap=None,
                 use_trial_cap=True, trial_cap=None,
                 initial_points_use_max=False,
                 random_probability=None, use_default_config=False,
                 initial_design="sobol", acquisition="ei", acquisition_xi=0.0,
                 challengers=None, local_search_iterations=None,
                 retrain_after=None,
                 rf_trees=None, rf_max_depth=None, rf_min_samples_split=None,
                 rf_min_samples_leaf=None, rf_feature_ratio=None,
                 rf_bootstrapping=True,
                 gp_model_type="vanilla", gp_restarts=None, gp_normalize_y=True):
        # `self._<name>` for every schema entry: the convention `get_params()`
        # reads back, and what makes the round trip through `.ihpo` work.
        self._search_strategy = search_strategy if search_strategy in _STRATEGIES else "gp"
        self._use_share_cap = bool(use_share_cap)
        self._share_cap = share_cap
        self._use_trial_cap = bool(use_trial_cap)
        self._trial_cap = trial_cap
        self._initial_points_use_max = bool(initial_points_use_max)
        self._random_probability = random_probability
        self._use_default_config = bool(use_default_config)
        self._initial_design = initial_design if initial_design in _INITIAL_DESIGNS else "sobol"
        self._acquisition = acquisition if acquisition in ("ei", "pi") else "ei"
        self._acquisition_xi = acquisition_xi
        self._challengers = challengers
        self._local_search_iterations = local_search_iterations
        self._retrain_after = retrain_after
        self._rf_trees = rf_trees
        self._rf_max_depth = rf_max_depth
        self._rf_min_samples_split = rf_min_samples_split
        self._rf_min_samples_leaf = rf_min_samples_leaf
        self._rf_feature_ratio = rf_feature_ratio
        self._rf_bootstrapping = bool(rf_bootstrapping)
        self._gp_model_type = gp_model_type if gp_model_type in _GP_MODELS else "vanilla"
        self._gp_restarts = gp_restarts
        self._gp_normalize_y = bool(gp_normalize_y)

    @classmethod
    def migrate_params(cls, stored: dict) -> dict:
        """Read the two settings the initial design used to have onto the four
        it has now.

        `exploration_ratio` was the share, and `exploration_trials` was an exact
        count that *replaced* it — it opened `max_ratio` right up so the share
        could not clamp it. So a stored count means the trial cap with the share
        cap switched off, which is the same search; a stored share alone means
        the share cap, with the per-hyperparameter bound still applying as it
        always did. Translated rather than dropped, because an experiment that
        came back configured differently from how it ran would say nothing about
        it.
        """
        if "exploration_ratio" not in stored and "exploration_trials" not in stored:
            return stored

        migrated = {k: v for k, v in stored.items()
                    if k not in ("exploration_ratio", "exploration_trials")}
        count = stored.get("exploration_trials")
        migrated.setdefault("share_cap", stored.get("exploration_ratio"))
        migrated.setdefault("use_share_cap", not count)
        migrated.setdefault("trial_cap", count)
        migrated.setdefault("use_trial_cap", True)
        return migrated

    def _initial_points(self, scenario, previous_result=None) -> int:
        """How many points to sample before the model takes over.

        **On a resume under a design whose points do not nest, the number the
        first run settled on wins.** The design is
        regenerated from scratch every run, and asking for a different size than
        last time asks for a different *set* of points: Sobol is a sequence, so
        a bigger draw contains the smaller one and SMAC skips what it has
        already evaluated, and the totals come out the same as an uninterrupted
        run. A Latin hypercube stratifies each dimension into `n` bins, so
        changing `n` moves every point and the whole design gets sampled again,
        in the middle of a search that had started modelling. Only the second
        kind is pinned; see `_EXTENDABLE_DESIGNS`.

        The size is not recomputed from what the history looks like, and not
        inferred from what the trials say about themselves. SMAC records it in
        the scenario, `deserialize_result` lifts it out, and this honours it.
        Recomputing would have to reconstruct an intent from origin strings
        written by another library for its own logs, and get two things right
        that it cannot: the model's own default configuration carries an
        initial-design origin while sitting outside `n_configs`, so counting
        origins grows the design by one on every resume; and a file written
        before origins were recorded labels every trial with the optimizer's
        name, which reads as "none of these were sampled".

        The recorded design is only honoured while it is still the design in
        use. Change the sampling method between runs and the number is
        re-derived, because keeping it would size a Latin hypercube draw by a
        Sobol count and resample everything — the opposite of the point.

        SMAC bounds this two ways and takes the smaller of them: a share of the
        budget (`max_ratio`) and a count (`n_configs`, or ten per hyperparameter
        when none is given). Both are exposed, either can be switched off, and
        the smaller can be made the larger — that last one is arithmetic here
        rather than a SMAC option, because SMAC only ever writes
        `min(n_configs, max_ratio * n_trials)`.

        Switching *both* off means neither bounds it, which is the whole budget:
        a search that only samples. A strange thing to ask for and a legible
        one, so it is allowed rather than quietly reinterpreted.

        The result is clamped to the budget, less the model's own defaults if
        those were asked for, because SMAC raises on an initial design that does
        not fit rather than truncating it — and to at least one, because a
        search has to start somewhere.
        """
        pinned = self._pinned_points(previous_result)
        if pinned is not None:
            return pinned

        budget = scenario.n_trials
        caps = []
        if self._use_share_cap:
            share = self._share_cap
            if share is None:
                share = _signature_default(AbstractInitialDesign.__init__, "max_ratio")
            caps.append(int(float(share) * budget))
        if self._use_trial_cap:
            count = self._trial_cap
            if count is None:
                count = _per_hyperparameter() * len(list(scenario.configspace.values()))
            caps.append(int(count))

        if not caps:
            wanted = budget
        else:
            wanted = max(caps) if self._initial_points_use_max else min(caps)

        room = budget - (1 if self._use_default_config else 0)
        return max(1, min(wanted, room))

    def _pinned_points(self, previous_result) -> int | None:
        """The size the initial design already settled on, or None to work it out.

        None whenever there is nothing to protect or nothing trustworthy to
        honour: a design whose points nest, a first run, a file from before
        SMAC's scenario was embedded, or a sampling method changed since.
        Falling through to the computed number is the safe direction in all of
        them — it is what every run did before this existed.
        """
        if self._initial_design in _EXTENDABLE_DESIGNS:
            # Asking for more points asks for the same points plus more, so
            # there is nothing to protect: let the phase grow with the budget,
            # which is what keeps a stopped-and-resumed experiment sampling as
            # much in total as an uninterrupted one.
            return None

        recorded = (previous_result.metadata.get("initial_design")
                    if previous_result is not None else None) or {}
        n_configs = recorded.get("n_configs")
        if n_configs is None:
            return None
        if recorded.get("name") != _INITIAL_DESIGNS[self._initial_design].__name__:
            return None
        return max(1, int(n_configs))

    @classmethod
    def strategy_defaults(cls) -> dict:
        """`{strategy: {param: default}}` for every setting whose default is the
        search strategy's rather than ours.

        The same lookup `resolved_params` does, exposed per strategy so a form
        can *show* what an empty field would mean instead of saying "search
        strategy default" and leaving the reader to go and find out. Both
        strategies are returned because the form's strategy selector changes
        client-side, and a placeholder that lied until the page reloaded would
        be worse than the vague wording it replaces.

        Read from SMAC's own signatures, not copied here — a copy is a second
        source of truth that goes stale without anything noticing.
        """
        return {
            name: {strategy: _signature_default(getattr(facade, getter), argument)
                   for strategy, facade in _STRATEGIES.items()}
            for name, (getter, argument) in _FACADE_DEFAULTS.items()
        }

    def resolved_params(self) -> dict:
        """Every setting with the blanks answered, for the record.

        A blank means "whatever this component already does", and what it does
        is a default sitting in a SMAC signature — read from there rather than
        copied here, because a copy is a second source of truth that goes stale
        without anything noticing. The version those signatures came from is in
        the file too, under `environment.packages.smac`.

        The blanks that stay blank are the ones that belong to the other search
        strategy, which has no default for them because it has no such
        component. Left as null: honest, and correct on re-import.
        """
        facade = _STRATEGIES[self._search_strategy]
        filled = dict(self.get_params())
        for name, (getter, argument) in _FACADE_DEFAULTS.items():
            if filled.get(name) is None:
                filled[name] = _signature_default(getattr(facade, getter), argument)
        if filled.get("share_cap") is None and self._use_share_cap:
            filled["share_cap"] = _signature_default(
                AbstractInitialDesign.__init__, "max_ratio")
        if filled.get("gp_restarts") is None and self._search_strategy == "gp":
            filled["gp_restarts"] = _signature_default(GaussianProcess.__init__, "n_restarts")
        return filled

    def _surrogate(self, facade, scenario):
        """The model the search fits, with whatever was set on it.

        Asked for through the facade's own `get_model` where that will take the
        setting, and built here where it will not: the Gaussian-process facade
        exposes only `model_type` and `kernel`, so restarts and target
        normalisation mean constructing the process directly. It is given the
        facade's own kernel, so the rest of the bundle still fits together — and
        with nothing set it is the same object `get_model` would have returned.

        Only settings that were actually given are passed, so an unset one keeps
        the strategy's default rather than one of ours.
        """
        if self._search_strategy == "rf":
            given = {
                "n_trees": self._rf_trees,
                "max_depth": self._rf_max_depth,
                "min_samples_split": self._rf_min_samples_split,
                "min_samples_leaf": self._rf_min_samples_leaf,
                # Above 1.0 SMAC computes `max_features = 0` and every split
                # considers no features at all — a forest of stumps, reported
                # as a fitted surrogate. The schema caps it and so does this.
                "ratio_features": (None if self._rf_feature_ratio is None
                                   else min(float(self._rf_feature_ratio), 1.0)),
            }
            return facade.get_model(
                scenario, bootstrapping=self._rf_bootstrapping,
                **{k: v for k, v in given.items() if v is not None})

        kernel = facade.get_kernel(scenario)
        if self._gp_model_type == "mcmc":
            # Mirrors the facade's own MCMC construction. The walker count is
            # derived from the kernel rather than chosen, and has to be even.
            walkers = 3 * len(kernel.theta)
            return MCMCGaussianProcess(
                configspace=scenario.configspace, kernel=kernel,
                n_mcmc_walkers=walkers + (walkers % 2),
                chain_length=250, burning_steps=250,
                normalize_y=self._gp_normalize_y, seed=scenario.seed)

        restarts = {} if self._gp_restarts is None else {"n_restarts": self._gp_restarts}
        return GaussianProcess(
            configspace=scenario.configspace, kernel=kernel,
            normalize_y=self._gp_normalize_y, seed=scenario.seed, **restarts)

    def _scenario_extras(self) -> dict:
        """Settings that belong to the scenario rather than to a component."""
        return {"use_default_config": self._use_default_config}

    def _facade(self, scenario, target_function, previous_result=None):
        """Build the chosen strategy, overriding only what was actually set.

        Each component is asked for through the facade's own `get_*`, so an
        unset knob keeps that strategy's default rather than one of ours. The
        two disagree — the Gaussian process scores a thousand candidates per
        trial and refits every one, the random forest scores ten thousand and
        refits every eighth — and neither number is right for the other.
        """
        facade = _STRATEGIES[self._search_strategy]

        design = _INITIAL_DESIGNS[self._initial_design]
        # The count is worked out here and handed over, rather than letting SMAC
        # derive it: `max_ratio` is opened up so it cannot clamp the number a
        # second time, and `n_configs` overrides the per-hyperparameter bound.
        # That is what makes the larger of the two caps reachable at all — see
        # `_initial_points`. `DefaultInitialDesign` is a single configuration
        # and ignores both.
        initial_design = design(
            scenario, max_ratio=1.0,
            n_configs=self._initial_points(scenario, previous_result))

        acquisition = facade.get_acquisition_function(scenario, xi=self._acquisition_xi)
        if self._acquisition == "pi":
            acquisition = PI(xi=self._acquisition_xi)

        model = self._surrogate(facade, scenario)
        # An MCMC process is an ensemble, and an acquisition function handed one
        # scores against whichever member it happens to hold. Marginalizing over
        # them is the whole point of sampling them, and nothing in SMAC pairs
        # these up for you — `IntegratedAcquisitionFunction` raises only if the
        # model has no members at all, so an unwrapped one fails quietly.
        if isinstance(model, MCMCGaussianProcess):
            acquisition = IntegratedAcquisitionFunction(acquisition)

        maximizer_kwargs = {}
        if self._challengers is not None:
            maximizer_kwargs["challengers"] = self._challengers
        if self._local_search_iterations is not None:
            maximizer_kwargs["local_search_iterations"] = self._local_search_iterations

        selector_kwargs = {}
        if self._retrain_after is not None:
            selector_kwargs["retrain_after"] = self._retrain_after

        random_kwargs = {}
        if self._random_probability is not None:
            random_kwargs["probability"] = self._random_probability

        return facade(
            scenario, target_function,
            model=model,
            initial_design=initial_design,
            acquisition_function=acquisition,
            acquisition_maximizer=facade.get_acquisition_maximizer(
                scenario, **maximizer_kwargs),
            random_design=facade.get_random_design(scenario, **random_kwargs),
            config_selector=facade.get_config_selector(scenario, **selector_kwargs),
            overwrite=True,
        )

    def _serialize_without_a_live_run(self, result: OptimizationResult) -> dict:
        """The base serialization, plus any SMAC state the result is carrying.

        Reached when there's no live output directory to read a runhistory from:
        either a run that never wrote one, or — the common case — a result
        rebuilt by `deserialize_result`, which now carries the embedded state in
        `metadata` rather than on disk.

        Without the pass-through, a load→save cycle would silently drop
        `optimizer_state` and with it the ability to resume an imported run.
        That used to be held together by `deserialize_result` writing the files
        out so this method could read them back; carrying the dict is the same
        round trip with the filesystem taken out of the middle.
        """
        carried = result.metadata.get("optimizer_state") or {}
        if not carried:
            return super().serialize_result(result)
        return {**super().serialize_result(result), "optimizer_state": carried}

    def serialize_result(self, result: OptimizationResult) -> dict:
        output_dir = result.metadata.get("smac_output_dir", "")
        root = Path(output_dir) if output_dir else None

        if not root or not root.exists():
            return self._serialize_without_a_live_run(result)

        rh_files = list(root.rglob("runhistory.json"))
        if not rh_files:
            return self._serialize_without_a_live_run(result)

        rh_path = rh_files[0]
        rh = json.loads(rh_path.read_text(encoding="utf-8"))

        trial_by_id = {t.trial: t for t in result.trials}
        config_key_to_id = {
            tuple(sorted(t.config.items())): str(t.trial) for t in result.trials
        }

        # Only trials this run actually evaluated. SMAC's runhistory is its own
        # bookkeeping and can hold entries we never recorded — a trial asked for
        # and not told because the run stopped, or one left behind in a reused
        # output directory. Copied out verbatim they became rows in the trials
        # table with no scores, which is a trial that never happened being
        # reported as one that scored nothing.
        recorded = []
        for entry in rh["data"]:
            t = trial_by_id.get(entry["config_id"])
            if t is None:
                continue
            entry["scores"] = t.scores
            entry["incumbent_score"] = t.incumbent_score
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            entry["incumbent_config_id"] = int(
                config_key_to_id.get(incumbent_key, str(entry["config_id"]))
            )
            recorded.append(entry)

        best_key = tuple(sorted(result.best_config.items())) if result.best_config else ()
        best_config_id = config_key_to_id.get(best_key) or (
            str(result.trials[-1].trial) if result.trials else "0"
        )

        optimizer_state = {
            f.relative_to(root).as_posix(): json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(root.rglob("*"))
            if f.is_file() and f != rh_path
        }

        return {
            # Counted from what was kept, not from what SMAC happened to hold.
            "stats": {"submitted": len(recorded), "finished": len(recorded), "running": 0},
            "data": recorded,
            "configs": rh["configs"],
            # Ours, keyed by trial number, not SMAC's, keyed by its own config
            # ids. Two reasons. The key spaces diverge as soon as a replayed
            # configuration is dropped or two trials share one configuration.
            # And SMAC reads origins off the live `Configuration` objects when
            # it saves, which the local-search maximizer relabels in place — so
            # its map says what the last local search touched, not where each
            # trial came from.
            "config_origins": {str(t.trial): t.origin for t in result.trials},
            "optimizer_state": optimizer_state,
            "primary_metric": result.primary_metric,
            "best_score": result.best_score,
            "best_config_id": best_config_id,
            "hyperparameter_importance": result.hyperparameter_importance,
            "hyperparameter_importance_warning": result.hyperparameter_importance_warning,
            "hyperparameter_sensitivity": result.hyperparameter_sensitivity,
            "hyperparameter_sensitivity_warning": result.hyperparameter_sensitivity_warning,
            "hyperparameter_mistunability": result.hyperparameter_mistunability,
            "hyperparameter_mistunability_warning": result.hyperparameter_mistunability_warning,
            "hyperparameter_interactions": result.hyperparameter_interactions,
            "hyperparameter_interactions_warning": result.hyperparameter_interactions_warning,
            "hyperparameter_moebius": result.hyperparameter_moebius,
            "hyperparameter_sensitivity_interactions": result.hyperparameter_sensitivity_interactions,
            "hyperparameter_sensitivity_moebius": result.hyperparameter_sensitivity_moebius,
            "hyperparameter_mistunability_interactions": result.hyperparameter_mistunability_interactions,
            "hyperparameter_mistunability_moebius": result.hyperparameter_mistunability_moebius,
            "hyperparameter_tunability_total": result.hyperparameter_tunability_total,
            "trials_limit": result.trials_limit,
        }

    def deserialize_result(self, d: dict) -> OptimizationResult:
        """Rebuild a result, carrying any embedded SMAC state in memory.

        This used to materialize `optimizer_state` into a fresh
        `tempfile.mkdtemp()` — every embedded file written back out, plus a
        reconstructed runhistory.json — on the theory that a resume would need
        it on disk. It doesn't, and nothing else does either:

        - `serialize_result` is the only reader of `metadata["smac_output_dir"]`,
          and it now takes the state straight from `metadata` instead.
        - A resume never reuses this directory. `optimize()` always makes its own
          (see the comment there) and replays history through `tell`.
        - The one thing a resume does lift out of stored state is
          `initial_design`, which is read from the in-memory dict just below.

        So the writes bought nothing and cost a directory per call — and since
        this runs on *every* rebuild (page load, trial click, ablation fetch,
        partial-dependence fetch), with nothing ever deleting them, they leaked
        one temp directory per request indefinitely.

        The keys are still validated even though nothing is written. Not writing
        makes traversal unreachable *here*, but the state is carried forward into
        whatever `serialize_result` writes next, and refusing a bad path at the
        boundary keeps it from being laundered through a new .ihpo. `io.parse`
        checks this too (`_check_optimizer_state`); this is the same guard for a
        result that reached the optimizer without passing through parse.
        """
        optimizer_state = d.get("optimizer_state", {})

        if not optimizer_state:
            return super().deserialize_result(d)

        for rel in optimizer_state:
            if not is_safe_relative(rel):
                raise ValueError(f"unsafe path in experiment file: {rel!r}")

        first_key = next(iter(optimizer_state))
        subdir = "/".join(first_key.split("/")[:-1])

        result = super().deserialize_result(d)
        result.metadata["optimizer_state"] = optimizer_state
        # How big the initial design was, and which design it was. SMAC records
        # both in the scenario it saves, and that file is already embedded here
        # verbatim — so the number a resume has to honour is in the .ihpo
        # without the .ihpo needing a field for it. See `_initial_points`.
        scenario = optimizer_state.get(f"{subdir}/scenario.json") or {}
        pinned = (scenario.get("_meta") or {}).get("initial_design")
        if isinstance(pinned, dict) and pinned.get("n_configs") is not None:
            result.metadata["initial_design"] = pinned
        return result

    def _confidence_nothing_better(self, smac, config_space, incumbent_config: dict):
        """How sure the surrogate is that no remaining configuration wins.

        This is what a Bayesian optimizer already knows and normally spends on
        choosing the next trial. The Gaussian process gives a posterior mean and
        standard deviation for any point in the space, so for a candidate *x*
        the chance it beats the incumbent is `Φ((μ(inc) − μ(x)) / σ(x))` — costs
        are minimized, so beating means lower. Take the best chance any sampled
        candidate has, and one minus it is the confidence that none of them does.

        The incumbent's *predicted* cost is the reference, not its measured one.
        A surrogate does not always predict in the units it was given — the
        random-forest strategy is trained on log-scaled costs — and comparing a
        measured cost against predictions in another space is not a comparison
        at all. Asking the model about both sides keeps the question inside
        whatever space it happens to think in, which is also how SMAC picks the
        reference for its own acquisition function (`_get_x_best`).

        Returns None when there is no answer to give: the model has not been
        fitted yet (SMAC trains it lazily, inside `ask`, and not before its
        initial design is done), or predicting failed. The criterion then simply
        does not fire, rather than a missing model reading as certainty.

        Two honest limits. It is confidence **under the surrogate's own model** —
        a GP with its own assumptions, and one that is poorly calibrated early
        on, which is why a handful of trials cannot end a run this way. And the
        space is *sampled*, not covered, so a narrow optimum that no sample lands
        near is not accounted for.
        """
        selector = getattr(smac.intensifier, "_config_selector", None)
        model = getattr(selector, "_model", None)
        if model is None:
            return None

        try:
            candidates = config_space.sample_configuration(_CONFIDENCE_SAMPLES)
            if not isinstance(candidates, list):
                candidates = [candidates]
            reference, _ = model.predict(
                convert_configurations_to_array(
                    [Configuration(config_space, values=incumbent_config)]),
                covariance_type="diagonal")
            # Variance, not standard deviation: the Gaussian process accepts
            # either, but the random forest raises on anything but "diagonal"
            # — inside the `except` below, which would have turned the whole
            # criterion off for that strategy without a word.
            mean, var = model.predict(
                convert_configurations_to_array(candidates), covariance_type="diagonal")
        except Exception:  # noqa: BLE001 — an unfitted or unhappy model means "no answer"
            return None
        if var is None:
            return None

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.sqrt(np.maximum(np.asarray(var, dtype=float).reshape(-1), 0.0))
        # A candidate the model is certain about beats the incumbent or does not;
        # the floor keeps that from dividing by zero on the way to saying so.
        z = (float(np.asarray(reference).reshape(-1)[0]) - mean) / np.maximum(std, 1e-12)
        best_chance = float(_normal_cdf(z).max())
        return 1.0 - best_chance

    def _replay(self, smac, config_space, trials, primary_metric: str, seed: int) -> None:
        """Tell SMAC an earlier run's trials, so a rebuilt surrogate knows them.

        Used when the stored SMAC state cannot be resumed — the optimized metric
        changed, or the run directory is gone. Without this the search would
        start blind while the page still showed the accumulated history.

        `save=False`: SMAC writes its runhistory on every `tell`, and rewriting
        a growing file once per replayed trial is pointless when the first real
        trial saves the lot. If the run is cancelled before any real trial, no
        runhistory is written and `serialize_result` falls back to rebuilding
        from the trials themselves, which is correct — just without SMAC state.
        """
        for t in trials:
            try:
                config = Configuration(config_space, values=t.config)
            except Exception:
                # A configuration the space no longer admits (the model's search
                # space changed under the experiment). Not tellable, and not
                # worth failing the run over.
                continue
            # Where it came from, put back before SMAC is told about it. A told
            # configuration with no origin is stamped "Custom" by SMAC, which is
            # how every replayed trial used to lose the one thing the record
            # wanted from it. Left as None when we never knew, so SMAC's own
            # label is at least honest about that.
            config.origin = t.origin or None
            info = t.run_info
            smac.tell(
                TrialInfo(config=config, seed=seed),
                TrialValue(
                    cost=1.0 - t.scores.get(primary_metric, t.score),
                    time=info.get("time", 0.0), cpu_time=info.get("cpu_time", 0.0),
                    starttime=info.get("starttime", 0.0), endtime=info.get("endtime", 0.0),
                    status=StatusType(info.get("status", STATUS_SUCCESS)),
                ),
                save=False,
            )

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None):
        # One fold unless the caller divided the data itself; see core.splits.
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)

        # A changed metric makes the stored surrogate worse than useless: it was
        # fitted on costs from a different objective, and resuming would mix the
        # two in one model. Rebuild instead, replaying the history below.
        previous_result, _ = rebase_history(previous_result, primary_metric)

        # Always a fresh directory, always rebuilt, always replayed. Resuming
        # SMAC's own stored state needed the scenario hash to be identical
        # between runs, which is why the budget below used to be a constant —
        # and that constant is what stopped the search ever reaching its model.
        # Replaying is a `tell` per past trial with no evaluation behind it,
        # which costs nothing beside a single model fit, and it removes the
        # `overwrite=False` path: on a mismatched directory SMAC asks the
        # *console* whether to continue, which in a worker is a hung run.
        trial_offset = len(previous_result.trials) if previous_result else 0
        output_dir = tempfile.mkdtemp()

        criteria = merge_stopping(n_trials, stopping)
        collector = TrialCollector(
            trial_offset=trial_offset,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=criteria,
        )

        config_space = model.get_config_space(seed=seed)
        scenario = Scenario(
            config_space,
            name="ihpo",
            # The real budget. SMAC sizes its initial design as a fraction of
            # this, so a budget of "effectively infinite" meant the fraction
            # never bit and every trial of a normal run came out of the initial
            # design — the model was fitted every iteration and never asked.
            n_trials=trial_offset + (criteria.get("max_trials") or _UNBOUNDED_BUDGET),
            deterministic=True,
            seed=seed,
            output_directory=Path(output_dir),
            **self._scenario_extras(),
        )

        def _unreachable(config, seed: int = 0) -> float:
            raise RuntimeError("SMAC called target_function unexpectedly in ask/tell mode")

        smac = self._facade(scenario, _unreachable, previous_result)

        wants_confidence = "incumbent_confidence" in criteria

        # A fresh facade knows nothing, so anything already evaluated has to be
        # handed to it — otherwise the search would begin from zero while the
        # page still shows the accumulated trials.
        if previous_result is not None:
            self._replay(smac, config_space, previous_result.trials, primary_metric, seed)

        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            info = smac.ask()
            config = dict(info.config)
            all_scores, run_info = evaluate_trial(model, config, splits, metrics, seed=seed)
            cost = 1.0 - all_scores[primary_metric]
            # Feed the measured timing into SMAC so its runhistory (which the
            # serialize override copies verbatim) carries the real values. The
            # status is passed through rather than assumed: a configuration the
            # model died on is something the surrogate should learn, not a
            # success that happened to score zero.
            smac.tell(info, TrialValue(
                cost=cost, time=run_info["time"], cpu_time=run_info["cpu_time"],
                starttime=run_info["starttime"], endtime=run_info["endtime"],
                status=StatusType(run_info["status"]),
                # Why it failed, or the runhistory records a zero with no
                # explanation and neither the page nor a later reader can say
                # whether the configuration was bad or the model was broken.
                additional_info=run_info.get("additional_info") or {},
            ))
            collector.record(config, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=info.config.origin or "")

            # Only worth asking if someone is listening, and only once there is
            # enough history for the answer to mean anything. The model is one
            # trial stale — SMAC trains it inside `ask` — which for a stopping
            # heuristic is close enough and avoids re-fitting it here.
            if wants_confidence and len(collector.results) >= _CONFIDENCE_MIN_TRIALS:
                collector.note_confidence(self._confidence_nothing_better(
                    smac, config_space, collector.incumbent_config))

        incumbent = smac.intensifier.get_incumbent()
        all_trials = (previous_result.trials if previous_result else []) + collector.results

        games = self.compute_hp_games(config_space, all_trials, metrics, seed=seed,
                                      cancel_event=cancel_event)

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=dict(incumbent) if incumbent else (all_trials[-1].config if all_trials else {}),
            # From `scores`, not `score`: after a metric change the history has
            # been re-read, and a trial that could not be (an old file with one
            # metric) must not contribute a score for a metric it never had.
            best_score=max((t.scores.get(primary_metric, t.score) for t in all_trials), default=0.0),
            hyperparameter_importance=games["tunability"][0],
            hyperparameter_importance_warning=games["tunability"][1],
            hyperparameter_sensitivity=games["sensitivity"][0],
            hyperparameter_sensitivity_warning=games["sensitivity"][1],
            hyperparameter_mistunability=games["mistunability"][0],
            hyperparameter_mistunability_warning=games["mistunability"][1],
            hyperparameter_interactions=games["tunability"][2],
            hyperparameter_interactions_warning=games["tunability"][1],
            hyperparameter_moebius=games["tunability"][3],
            hyperparameter_sensitivity_interactions=games["sensitivity"][2],
            hyperparameter_sensitivity_moebius=games["sensitivity"][3],
            hyperparameter_mistunability_interactions=games["mistunability"][2],
            hyperparameter_mistunability_moebius=games["mistunability"][3],
            hyperparameter_tunability_total=games["tunability"][4],
            metadata={"smac_output_dir": str(output_dir),
                      "stopped_by": collector.stopped_by},
        )
