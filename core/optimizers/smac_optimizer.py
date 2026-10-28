import json
import logging
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ConfigSpace import Configuration
from scipy.stats import norm
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import PI, IntegratedAcquisitionFunction
from smac.initial_design import (
    AbstractInitialDesign, DefaultInitialDesign, LatinHypercubeInitialDesign,
    RandomInitialDesign, SobolInitialDesign,
)
from smac.main.exceptions import ConfigurationSpaceExhaustedException
from smac.model.gaussian_process import GaussianProcess, MCMCGaussianProcess
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.utils.configspace import convert_configurations_to_array

from ..analytics import compute_hp_games, pair_trials_with_scores
from ..metrics import metric_for, to_cost
from ..paths import is_safe_relative
from ..splits import holdout
from .. import priors as priors_module
from .base import (
    REQUESTED_ORIGIN, STOPPED_BY_EXHAUSTED, BaseOptimizer, game_fields, merge_stopping,
    OptimizationResult, OptimizerParam, rebase_history, storable_score,
)
from .timing import STATUS_SUCCESS
from .trial import evaluate_trial

logging.getLogger("smac").setLevel(logging.WARNING)

#: A prior that fails to apply must log instead of raising or staying silent.
logger = logging.getLogger(__name__)

#: `SLICE_CANDIDATES` is also the extra budget of a reconstructed scenario, so
#: the figure quotes the decay factor the walk uses.
SLICE_CANDIDATES = 12

#: `TabulatedPrior` coarsens the grid anyway, so 512 points only need to catch
#: the editor's narrowest peak, σ = 0.004.
_PRIOR_GRID = 512


def decay_beta(n_trials) -> float:
    """The search and the figure share `decay_beta`, so a caption cannot drift
    from the prior's decay. The factor comes from the budget, since a
    finished-trial count would grow and stop the decay.
    """
    return max(float(n_trials or 0) / 10.0, 1.0)

# `_UNBOUNDED_BUDGET` stands in for a missing trial cap and only sizes the
# initial design, where the share cap is the real setting.
_UNBOUNDED_BUDGET = 100


def _when(report, smac):
    """`_when` stamps *report* with the trial count and time the prior took
    effect, and passes None through."""
    if report is None:
        return None
    from datetime import datetime, timezone

    return {**report, "at_trial": len(smac.runhistory),
            "at": datetime.now(timezone.utc).isoformat()}


def _waits_for_the_design(priors) -> bool:
    """`_apply_priors` puts every belief into one weight, so one waiting belief
    delays them all.
    """
    return any(bool((stated or {}).get("delay_decay")) for stated in (priors or {}).values())


def scenario_budget(trial_offset, max_trials=None) -> int:
    """`scenario_budget` serves a run, its re-walk and the figure, which
    therefore quote the same decay factor.
    """
    return int(trial_offset) + int(max_trials or _UNBOUNDED_BUDGET)

#: Per hyperparameter, because the open space grows with each one.
PRIOR_JUDGING_SAMPLES_PER_HYPERPARAMETER = 1000

#: Each facade bundles a surrogate, acquisition function, maximizer and encoder
#: chosen to work together.
STRATEGIES = {
    "gp": BlackBoxFacade,
    "rf": HyperparameterOptimizationFacade,
}

#: The `mcmc` fit draws kernel hyperparameters as an ensemble, which is better
#: calibrated and about ten times slower than the single likelihood fit of
#: `vanilla`.
_GP_MODELS = ("vanilla", "mcmc")

#: `INITIAL_POINTS` names the form group for the two initial-design caps and the
#: choice of the binding cap, since all three set how many starting points SMAC
#: samples.
INITIAL_POINTS = "initial_points"


#: A Sobol or seeded random draw is a sequence, and a resumed run asking for
#: more points gets the old ones plus the rest. A Latin hypercube shifts every
#: point when `n` changes, so a resume under it keeps its size.
_EXTENDABLE_DESIGNS = frozenset({"sobol", "random", "default_only"})

INITIAL_DESIGNS = {
    "sobol": SobolInitialDesign,
    "latin_hypercube": LatinHypercubeInitialDesign,
    "random": RandomInitialDesign,
    "default_only": DefaultInitialDesign,
}

# Each confidence check makes a thousand surrogate predictions, which take
# milliseconds beside a trial.
_CONFIDENCE_SAMPLES = 1000

# A GP fitted on a handful of points looks confident only because nothing
# contradicts it.
_CONFIDENCE_MIN_TRIALS = 10


#: `_FACADE_DEFAULTS` reads each blank setting's default from the installed SMAC
#: facade signature, which keeps the export current. The `rf_` entries resolve
#: only under `rf`, and `_signature_default` returns None otherwise.
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


def _signature_default(function, argument) -> Any:
    """`_signature_default` looks up the default of *argument*, or None."""
    import inspect

    parameter = inspect.signature(function).parameters.get(argument)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


class _RecordedWalk(list):
    """`LocalSearch._search` replaces list entries as it climbs, so
    `_RecordedWalk` records each entry's whole path with SMAC unchanged.
    """

    def __init__(self, start_points, sink):
        super().__init__(start_points)
        self.paths = [[point] for point in start_points]
        sink.extend(self.paths)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        path = self.paths[index]
        if path[-1] != value:
            path.append(value)


def _record_walks(smac) -> list:
    """`_record_walks` collects every later local-search path and pins the
    search to one process, since worker paths never return."""
    selector = getattr(smac.intensifier, "_config_selector", None)
    maximizer = getattr(selector, "_acquisition_maximizer", None)
    local = getattr(maximizer, "_local_search", maximizer)
    paths: list = []
    if local is None or not hasattr(local, "_search"):
        return paths
    search = local._search
    local.n_jobs = 1

    def recording(start_points):
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        return search(_RecordedWalk(start_points, paths))

    local._search = recording
    return paths


def _walks_to(asked, paths) -> list:
    """`_walks_to` finds, for each asked configuration, the unused local-search
    path that ended at it, or None when no climb reached it."""
    used: set = set()
    out = []
    for info in asked:
        match = None
        for index, path in enumerate(paths):
            if index not in used and len(path) and path[-1] == info.config:
                match = index
                break
        if match is None:
            out.append(None)
            continue
        used.add(match)
        out.append([dict(point) for point in paths[match]])
    return out


class _RunSurrogate:
    """The model learned cost, so `from_cost` converts the mean and leaves the
    variance, which a negation or subtraction keeps.
    """

    def __init__(self, model, metric):
        self._model = model
        self._metric = metric
        #: A GP is smooth while a forest is a staircase, and a picture may be
        #: interpolated only for a GP.
        self.smooth = "Gaussian" in type(model).__name__

    def predict_marginalized(self, rows):
        """`predict_marginalized` returns `(mean, variance)`."""
        import numpy as np

        from core.metrics import from_cost

        mean, var = self._model.predict_marginalized(np.asarray(rows))
        mean = np.asarray(mean, dtype=float).ravel()
        if self._metric is not None:
            mean = np.array([from_cost(self._metric, float(m)) for m in mean])
        return mean, np.asarray(var, dtype=float).ravel()


class SMACOptimizer(BaseOptimizer):
    """Unset settings take SMAC's default for the chosen strategy, since the two
    strategies disagree on several.
    """

    name = "SMAC"

    #: `aliases` keeps the name from when SMAC always used the Gaussian-process
    #: facade.
    aliases = ("SMAC (BlackBox)",)

    #: The surrogate can report its confidence (see `incumbent_confidence`).
    supports_confidence_stopping = True

    #: A changed metric invalidates the surrogate carried between runs.
    fits_surrogate = True

    def acquisition_function(self):
        return {"kind": self._acquisition, "xi": float(self._acquisition_xi or 0.0)}

    @classmethod
    def judges_priors(cls, params):
        """`judges_priors` accepts only the GP, because the forest's log-scaled
        lower confidence bound maps to near-zero cost everywhere and every prior
        would pass."""
        strategy = cls.known_params(params).get("search_strategy")
        return (strategy if strategy in STRATEGIES else "gp") == "gp"

    params_schema = [
        OptimizerParam("search_strategy", "Search strategy", "select", "gp",
                       choices=["gp", "rf"]),
        # SMAC takes the smaller of a budget share and a count, and both caps
        # can be switched off or swapped to the larger (see `_initial_points`).
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

        # Every surrogate setting carries the `rf_` prefix and a "surrogate"
        # label, because the Random Forest model's own `max_depth` and
        # `min_samples_split` can be on screen at once.
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
        # Above 1.0 SMAC computes `max_features = 0`.
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
        # `get_params()` reads these back for the `.ihpo` round trip.
        self._search_strategy = search_strategy if search_strategy in STRATEGIES else "gp"
        self._use_share_cap = bool(use_share_cap)
        self._share_cap = share_cap
        self._use_trial_cap = bool(use_trial_cap)
        self._trial_cap = trial_cap
        self._initial_points_use_max = bool(initial_points_use_max)
        self._random_probability = random_probability
        self._use_default_config = bool(use_default_config)
        self._initial_design = initial_design if initial_design in INITIAL_DESIGNS else "sobol"
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
    def migrate_params(cls, stored: Mapping[str, Any]) -> Mapping[str, Any]:
        """`exploration_trials` was an exact count that bypassed the share, so
        it becomes the trial cap with the share cap off. `exploration_ratio`
        alone becomes the share cap.
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
        """A resume under a non-nesting design such as Latin hypercube keeps the
        size the first run recorded (see `_pinned_points`), since a new size
        resamples every point mid-search. SMAC takes the smaller of a budget
        share (`max_ratio`) and a count (`n_configs`, ten per hyperparameter by
        default), and choosing the larger is arithmetic here. With both caps off
        the whole budget is sampled. The result is clamped to at least one and
        to the budget less the default configuration, because SMAC raises on an
        oversized design.
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
                # SMAC's per-hyperparameter count, read from its signature.
                per_hp = _signature_default(AbstractInitialDesign.__init__,
                                            "n_configs_per_hyperparameter") or 10
                count = per_hp * len(list(scenario.configspace.values()))
            caps.append(int(count))

        if not caps:
            wanted = budget
        else:
            wanted = max(caps) if self._initial_points_use_max else min(caps)

        room = budget - (1 if self._use_default_config else 0)
        return max(1, min(wanted, room))

    def _pinned_points(self, previous_result) -> int | None:
        """`_pinned_points` returns None for a nesting design, a first run, a
        file without SMAC's scenario, or a changed sampling method.
        """
        if self._initial_design in _EXTENDABLE_DESIGNS:
            # A nesting design lets the phase grow with the budget, so a resumed
            # run samples as much as an uninterrupted one.
            return None

        recorded = (previous_result.metadata.get("initial_design")
                    if previous_result is not None else None) or {}
        n_configs = recorded.get("n_configs")
        if n_configs is None:
            return None
        if recorded.get("name") != INITIAL_DESIGNS[self._initial_design].__name__:
            return None
        return max(1, int(n_configs))

    @classmethod
    def strategy_defaults(cls) -> dict:
        """Both strategies' defaults come from SMAC's signatures, because the
        form switches strategy client-side and its placeholders must stay
        correct.
        """
        return {
            name: {strategy: _signature_default(getattr(facade, getter), argument)
                   for strategy, facade in STRATEGIES.items()}
            for name, (getter, argument) in _FACADE_DEFAULTS.items()
        }

    def resolved_params(self) -> dict:
        """Blank settings take SMAC's signature defaults, and the SMAC version
        sits in `environment.packages.smac`. Settings of the other strategy stay
        null, which re-imports correctly.
        """
        facade = STRATEGIES[self._search_strategy]
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

    def refit_surrogate(self, config_space, trials, metric_name: str, seed: int = 0,
                        values=None):
        """A GP run is read through a GP, since a generic forest would make
        every surrogate look alike. The model trains on cost and answers in
        metric units, or on *values* as given, and a failed refit returns
        `(None, warning)`.
        """
        data = pair_trials_with_scores(config_space, trials, metric_name, values=values)
        if len(data) < 2:
            return None, "Not enough trials to fit a surrogate."

        metric = None if values is not None else metric_for(metric_name)
        X = np.array([config.get_array() for config, _ in data])
        y = np.array([float(target) if metric is None else to_cost(metric, target)
                      for _, target in data])

        try:
            with tempfile.TemporaryDirectory() as directory:
                scenario = Scenario(config_space, n_trials=max(len(y), 1), seed=seed,
                                    deterministic=True, output_directory=Path(directory))
                model = self._surrogate(STRATEGIES[self._search_strategy], scenario,
                                        log_y=False)
                model.train(X, y)
        except Exception as error:  # noqa: BLE001 — any refit failure falls back
            return None, f"Could not rebuild this run's surrogate: {error}"

        return _RunSurrogate(model, metric), None

    def _surrogate(self, facade, scenario, log_y=None):
        """The GP facade exposes only `model_type` and `kernel`, so restarts and
        normalisation require building the process here with the facade's
        kernel. Only given settings are passed, so an unset one keeps the
        strategy's default.
        """
        if self._search_strategy == "rf":
            given = {
                "n_trees": self._rf_trees,
                "max_depth": self._rf_max_depth,
                "min_samples_split": self._rf_min_samples_split,
                "min_samples_leaf": self._rf_min_samples_leaf,
                # SMAC computes `max_features = 0` above 1.0.
                "ratio_features": (None if self._rf_feature_ratio is None
                                   else min(float(self._rf_feature_ratio), 1.0)),
            }
            given = {k: v for k, v in given.items() if v is not None}
            if log_y is None:
                return facade.get_model(
                    scenario, bootstrapping=self._rf_bootstrapping, **given)
            # `get_model` hardcodes `log_y=True`, and raw costs can be negative,
            # so a refit builds the forest here with `log_y` off.
            from smac.model.random_forest.random_forest import RandomForest
            defaults = dict(n_trees=10, ratio_features=1.0, min_samples_split=2,
                            min_samples_leaf=1, max_depth=2 ** 20)
            return RandomForest(
                log_y=log_y, bootstrapping=self._rf_bootstrapping,
                **{**defaults, **given},
                configspace=scenario.configspace,
                instance_features=scenario.instance_features, seed=scenario.seed)

        kernel = facade.get_kernel(scenario)
        if self._gp_model_type == "mcmc":
            # `walkers` mirrors the facade's MCMC setup and must be even.
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
        """`_scenario_extras` adds the use-default-config switch to scenarios."""
        return {"use_default_config": self._use_default_config}

    @contextmanager
    def _rebuilt(self, config_space, trials, metric_name: str, seed: int, previous_result,
                 n_trials: int, purpose: str):
        """`_rebuilt` provides a run's facade with *trials* replayed, in a
        scratch directory, and its target raises, naming *purpose*."""
        # The recorded design size, as `_pinned_points` reads it.
        recorded = ((getattr(previous_result, "metadata", None) or {})
                    .get("initial_design") or {}).get("n_configs")
        with tempfile.TemporaryDirectory() as directory:
            scenario = Scenario(
                config_space, name="ihpo", n_trials=n_trials, deterministic=True, seed=seed,
                output_directory=Path(directory), **self._scenario_extras())

            def _unreachable(config, seed: int = 0) -> float:
                raise RuntimeError(f"target_function called while {purpose}")

            smac = self._facade(scenario, _unreachable, None, initial_points=recorded)
            self._replay(smac, config_space, trials, metric_name, seed)
            yield smac, scenario

    def slice_challengers(self, config_space, trials, metric_name: str, seed: int = 0,
                          priors=None, count=SLICE_CANDIDATES, previous_result=None,
                          budget=None, with_walks=False):
        """`slice_challengers` builds the facade with `_facade`, replays the
        trials, applies priors and calls `ask()`, returning `(configs, origins)`
        in the search's own rank order, or None. The initial design keeps the
        size the run recorded, since a rebuild sized for an unbounded budget
        would propose space-filling samples. Successive calls to `ask()` without
        a `tell()` walk the maximizer's cached challenger list. *with_walks*
        adds each candidate's local-search path (see `_record_walks`), or None
        for one reached without climbing.
        """
        if not trials:
            return None
        try:
            with self._rebuilt(config_space, trials, metric_name, seed, previous_result,
                               len(trials) + max(count, 1), "ranking candidates") as (smac, scenario):
                self._apply_priors(smac, scenario, config_space, priors, budget=budget)
                walks = _record_walks(smac) if with_walks else None
                asked = [smac.ask() for _ in range(count)]
            configs = [dict(info.config) for info in asked]
            origins = [info.config.origin or "" for info in asked]
            if walks is None:
                return configs, origins
            return configs, origins, _walks_to(asked, walks)
        except Exception as error:  # noqa: BLE001 — a diagnostic must not cost the figure
            logger.warning("Could not ask this run's optimizer what is next: %s", error)
            return None

    def evaluate_prior(self, config_space, trials, metric_name: str, seed: int = 0,
                       priors=None, tolerance: float = 0.15, previous_result=None,
                       budget=None, top_k: int = 10):
        """SMAC's `ClimbingComparisonPolicy` runs DynaBO's safeguard on demand,
        climbing from the best *top_k* draws of the belief and of the
        incumbent's neighbourhood and rejecting the belief when its climbs end
        worse by more than *tolerance*. The facade is rebuilt as in
        `slice_challengers`, and the verdict comes from `add_prior`. *tolerance*
        is positive, in raw objective units, and passed negated as the policy
        expects. An "unjudged" verdict means the policy had nothing to compare
        and accepted by rule, which `_RecordingPolicy` detects.
        """
        if not priors:
            return {"verdict": "unjudged", "reason": "no prior is stated"}
        if not trials:
            return {"verdict": "unjudged",
                    "reason": "nothing has been run yet, so there is no model to "
                              "judge this against"}
        try:
            from smac.acquisition.weight import ClimbingComparisonPolicy
        except ImportError:
            return {"verdict": "unjudged",
                    "reason": "this SMAC has no climbing comparison policy"}

        class _RecordingPolicy(ClimbingComparisonPolicy):
            compared = None

            def compare(self, *args, **kwargs):
                self.compared = super().compare(*args, **kwargs)
                return self.compared

        try:
            # DynaBO draws the neighbourhood as wide as the belief.
            policy = _RecordingPolicy(
                threshold=-abs(tolerance), top_k=int(top_k),
                n_samples_per_hyperparameter=PRIOR_JUDGING_SAMPLES_PER_HYPERPARAMETER,
                neighbourhood_std="prior")
            with self._rebuilt(config_space, trials, metric_name, seed, previous_result,
                               len(trials) + 1, "judging a prior") as (smac, scenario):
                applied = self._apply_priors(
                    smac, scenario, config_space, priors, budget=budget, acceptance=policy)
        except Exception as error:  # noqa: BLE001 — a diagnostic must not cost the figure
            logger.warning("Could not judge the stated prior: %s", error)
            return {"verdict": "unjudged", "reason": str(error)}

        if applied is None:
            return {"verdict": "unjudged", "reason": "no prior is stated"}
        if applied.get("applied") and policy.compared is None:
            return {"verdict": "unjudged",
                    "reason": "there is no model or incumbent to judge this against"}
        if applied.get("applied"):
            return {"verdict": "accepted",
                    "reason": "the surrogate does not think this region is worse "
                              "than where the search is already looking"}
        if applied.get("rejected"):
            return {"verdict": "rejected", "reason": applied.get("reason", "")}
        # A prior that failed to apply is no judgement on the belief.
        return {"verdict": "unjudged", "reason": applied.get("reason", "")}

    @staticmethod
    def _prior_grid(hyperparameter):
        """A prior is evaluated against vectorized values and stated on the unit
        interval, which coincide for a numerical hyperparameter. A categorical
        is evaluated only at its choices, placed evenly along the stated axis.
        """
        choices = getattr(hyperparameter, "choices", None)
        if choices:
            vectors = [float(hyperparameter.to_vector(c)) for c in choices]
            last = max(len(choices) - 1, 1)
            return vectors, [i / last for i in range(len(choices))]
        grid = [i / (_PRIOR_GRID - 1) for i in range(_PRIOR_GRID)]
        return grid, grid

    def _apply_priors(self, smac, scenario, config_space, priors, budget=None,
                      acceptance=None):
        """*priors* maps hyperparameters to stated `kind`, `params` and `decay`,
        and `core.priors.density_from` evaluates each density here. All beliefs
        enter as one `TabulatedPrior` over the whole space, since
        `validate_against` needs every hyperparameter in column order and
        separate priors would multiply as an ensemble. A prior that fails is
        reported and skipped, and the returned report goes onto the result's
        metadata so a file records whether the prior took.
        """
        if not priors:
            return None
        try:
            from smac.acquisition.weight import TabulatedPrior, get_decay_schedule
        except ImportError:
            logger.warning("This SMAC has no TabulatedPrior, so stated priors are ignored. "
                           "codesigner needs the branch carrying the acquisition weight layer.")
            return {"applied": False,
                    "reason": "this SMAC has no acquisition weight layer"}

        tables, shape, stated_beta, stated_ratio = {}, "none", None, None
        statements = []
        for name, stated in (priors or {}).items():
            if name not in config_space:
                continue
            # The figure gets its density from the same `core.priors` function
            # through `prior_density`.
            vectors, units = self._prior_grid(config_space[name])
            values = priors_module.density_from(
                (stated or {}).get("kind") or "uniform",
                (stated or {}).get("params") or {}, units) or []
            if len(values) >= 2:
                # The table uses vector positions, while the density came from
                # the unit interval.
                tables[name] = [[v, y] for v, (_, y) in zip(vectors, values)]
                statements.append(stated or {})

        # One weight carries one schedule, so the last named belief settles it.
        for stated in statements:
            decay = stated.get("decay")
            if isinstance(decay, dict):
                shape = decay.get("shape") or "none"
                # πBO and DynaBO state β as a budget ratio, and an absolute
                # `beta` still loads for older priors.
                if decay.get("beta_ratio") is not None:
                    stated_ratio = float(decay["beta_ratio"])
                elif decay.get("beta") is not None:
                    stated_beta = float(decay["beta"])
            else:
                shape = decay or "none"
        if not tables:
            return {"applied": False,
                    "reason": "no stated prior named a hyperparameter of this "
                              "search space, or none described a curve"}

        # The decay factor comes from *budget*, the run's configured budget,
        # because a re-walk's scenario is sized to its replayed trials. A stated
        # ratio resolves against that budget, and `decay_beta` (N/10, as in πBO
        # and DynaBO) stands in otherwise.
        run_budget = (budget if budget is not None
                      else getattr(scenario, "n_trials", 0))
        if stated_ratio is not None:
            beta = max(float(run_budget or 0) * stated_ratio, 1e-6)
        elif stated_beta is not None:
            beta = stated_beta
        else:
            beta = decay_beta(run_budget)

        try:
            prior = TabulatedPrior(config_space, tables)
            prior.validate_against(config_space)
            # `add_prior` returns None when the acceptance policy refuses the
            # belief, and the default policy accepts everything.
            key = smac.add_prior(prior, key="codesigner",
                                 decay=get_decay_schedule(shape, beta),
                                 acceptance_policy=acceptance)
        except Exception as error:  # noqa: BLE001 — a prior must not cost a run
            logger.warning("Could not apply the stated prior, continuing without it: %s", error)
            return {"applied": False, "reason": str(error)}

        if key is None:
            # A refusal is the only failure that judges the reader's belief, so
            # the report marks it.
            return {"applied": False, "rejected": True,
                    "hyperparameters": sorted(tables), "decay": shape, "beta": beta,
                    "reason": "the surrogate scores this region worse than the "
                              "incumbent's neighbourhood by more than the "
                              "tolerance allows"}

        return {"applied": True, "key": key, "hyperparameters": sorted(tables),
                "decay": shape, "beta": beta}

    def _facade(self, scenario, target_function, previous_result=None,
                initial_points=None):
        """Each component comes from the facade's own `get_*`, so an unset knob
        keeps the strategy's default, which differs between GP and forest.
        """
        facade = STRATEGIES[self._search_strategy]

        design = INITIAL_DESIGNS[self._initial_design]
        # `max_ratio=1.0` and an explicit `n_configs` stop SMAC clamping the
        # count again, which makes the larger cap reachable (see
        # `_initial_points`). *initial_points* lets a rebuild reuse a finished
        # run's design size instead of guessing from a budget.
        initial_design = design(
            scenario, max_ratio=1.0,
            n_configs=(initial_points if initial_points is not None
                       else self._initial_points(scenario, previous_result)))

        acquisition = facade.get_acquisition_function(scenario, xi=self._acquisition_xi)
        if self._acquisition == "pi":
            acquisition = PI(xi=self._acquisition_xi)

        model = self._surrogate(facade, scenario)
        # An acquisition function handed an MCMC ensemble scores one member, so
        # `IntegratedAcquisitionFunction` marginalizes over all of them.
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

    def serialize_result(self, result: OptimizationResult) -> dict:
        """SMAC's runhistory and output directory replace the base trial table
        when present."""
        out = super().serialize_result(result)
        output_dir = result.metadata.get("smac_output_dir", "")
        root = Path(output_dir) if output_dir else None
        rh_files = list(root.rglob("runhistory.json")) if root and root.exists() else []
        if root is None or not rh_files:
            # A result rebuilt by `deserialize_result` carries its state in
            # `metadata`, and passing that through lets an imported run resume
            # after a save.
            carried = result.metadata.get("optimizer_state") or {}
            return {**out, "optimizer_state": carried} if carried else out

        rh_path = rh_files[0]
        rh = json.loads(rh_path.read_text(encoding="utf-8"))

        trial_by_id = {t.trial: t for t in result.trials}
        config_key_to_id = {
            tuple(sorted(t.config.items())): str(t.trial) for t in result.trials
        }

        # SMAC's runhistory can hold entries the run never recorded, such as a
        # trial asked for before a stop, and those would appear as unscored
        # trials.
        recorded = []
        for entry in rh["data"]:
            t = trial_by_id.get(entry["config_id"])
            if t is None:
                continue
            entry["scores"] = t.scores
            entry["incumbent_score"] = storable_score(t.incumbent_score)
            incumbent_key = tuple(sorted(t.incumbent_config.items()))
            entry["incumbent_config_id"] = int(
                config_key_to_id.get(incumbent_key, str(entry["config_id"]))
            )
            recorded.append(entry)

        return {
            **out,
            # Counts follow the kept entries.
            "stats": {"submitted": len(recorded), "finished": len(recorded), "running": 0},
            "data": recorded,
            "configs": rh["configs"],
            "optimizer_state": {
                f.relative_to(root).as_posix(): json.loads(f.read_text(encoding="utf-8"))
                for f in sorted(root.rglob("*"))
                if f.is_file() and f != rh_path
            },
        }

    def deserialize_result(self, d: dict) -> OptimizationResult:
        """The SMAC state stays in `metadata`, since a resume makes its own
        directory and replays history through `tell`. State keys are validated
        anyway, so a bad path cannot pass through into the next `.ihpo`.
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
        # SMAC records the initial design in its embedded scenario.json, which a
        # resume honours (see `_initial_points`).
        scenario = optimizer_state.get(f"{subdir}/scenario.json") or {}
        pinned = (scenario.get("_meta") or {}).get("initial_design")
        if isinstance(pinned, dict) and pinned.get("n_configs") is not None:
            result.metadata["initial_design"] = pinned
        return result

    def _confidence_nothing_better(self, smac, config_space, incumbent_config: Optional[dict]):
        """For each sampled candidate the chance of beating the incumbent is
        `Φ((μ(inc) − μ(x)) / σ(x))`, and the confidence is one minus the best
        chance. The incumbent's predicted cost is the reference, since the
        forest predicts log-scaled costs. None means an unfitted model or a
        failed prediction, and the answer holds only under the surrogate's own
        model over a sample of the space.
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
            # The forest raises on any covariance type except "diagonal", which
            # would disable the criterion silently.
            mean, var = model.predict(
                convert_configurations_to_array(candidates), covariance_type="diagonal")
        except Exception:  # noqa: BLE001 — an unfitted or unhappy model means "no answer"
            return None
        if var is None:
            return None

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.sqrt(np.maximum(np.asarray(var, dtype=float).reshape(-1), 0.0))
        # The floor avoids dividing by zero for a certain candidate.
        z = (float(np.asarray(reference).reshape(-1)[0]) - mean) / np.maximum(std, 1e-12)
        best_chance = float(norm.cdf(z).max())
        return 1.0 - best_chance

    def _replay(self, smac, config_space, trials, primary_metric: str, seed: int,
                replay_metric=None) -> None:
        """`_replay` tells SMAC an earlier run's trials when stored state cannot
        resume, after a metric change or a lost directory. `save=False` skips
        rewriting the runhistory per replayed trial, since the first real trial
        saves the file.
        """
        replay_metric = replay_metric or metric_for(primary_metric)
        for t in trials:
            try:
                config = Configuration(config_space, values=t.config)
            except Exception:
                # The search space changed, so the configuration cannot be told.
                continue
            # SMAC stamps a told configuration that lacks an origin as Custom,
            # so the stored origin goes back first, with None when unknown.
            config.origin = t.origin or None
            info = t.run_info
            smac.tell(
                TrialInfo(config=config, seed=seed),
                TrialValue(
                    # `to_cost` gives older metrics `1.0 - score`, and a replay
                    # therefore tells SMAC the first run's costs.
                    cost=to_cost(replay_metric, t.scores.get(primary_metric, t.score)),
                    time=info.get("time", 0.0), cpu_time=info.get("cpu_time", 0.0),
                    starttime=info.get("starttime", 0.0), endtime=info.get("endtime", 0.0),
                    status=StatusType(info.get("status", STATUS_SUCCESS)),
                ),
                save=False,
            )

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None, priors=None,
                 configurations=None):
        # Holdout unless the caller gave splits.
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)

        # A surrogate fitted to another objective must be rebuilt, so the
        # history is replayed below.
        previous_result, _ = rebase_history(previous_result, primary_metric)

        # Each run uses a fresh directory and replays history, which costs
        # little and avoids SMAC's `overwrite=False` console prompt that hangs a
        # worker.
        trial_offset = len(previous_result.trials) if previous_result else 0
        output_dir = tempfile.mkdtemp()

        criteria = merge_stopping(n_trials, stopping)
        collector = self.new_collector(
            metric_name=primary_metric,
            previous_result=previous_result,
            trial_offset=trial_offset,
            # None means the metric's own worst end, +inf for lower-is-better.
            initial_best_score=previous_result.best_score if previous_result else None,
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=criteria,
        )

        config_space = model.get_config_space(seed=seed)
        scenario = Scenario(
            config_space,
            name="ihpo",
            # SMAC sizes its initial design as a fraction of the real budget.
            n_trials=scenario_budget(trial_offset, criteria.get("max_trials")),
            deterministic=True,
            seed=seed,
            output_directory=Path(output_dir),
            **self._scenario_extras(),
        )

        def _unreachable(config, seed: int = 0) -> float:
            raise RuntimeError("SMAC called target_function unexpectedly in ask/tell mode")

        smac = self._facade(scenario, _unreachable, previous_result)

        wants_confidence = "incumbent_confidence" in criteria

        # A fresh facade needs the evaluated trials to avoid starting from zero.
        if previous_result is not None:
            self._replay(smac, config_space, previous_result.trials, primary_metric,
                         seed, replay_metric=collector.metric)

        # `add_prior` anchors the decay at the finished trials it sees and
        # refuses to re-anchor, so the prior goes in after the replay. A belief
        # that waits goes in once the initial design is exhausted, since the
        # design trials are beyond its reach and fading through them would
        # weaken it.
        waiting = _waits_for_the_design(priors)
        # The design end is an absolute count from the start of the search,
        # since a resume continues inside the design.
        design_end = (self._initial_points(scenario, previous_result)
                      if waiting else 0)
        prior_report = None
        if not waiting:
            prior_report = _when(self._apply_priors(smac, scenario, config_space, priors), smac)

        # Requested configurations are told to SMAC like its own, since the
        # model learns the same from them.
        def tell_requested(config, all_scores, run_info):
            chosen = Configuration(config_space, values=config)
            chosen.origin = REQUESTED_ORIGIN
            smac.tell(TrialInfo(config=chosen, seed=seed), TrialValue(
                cost=to_cost(collector.metric, all_scores[primary_metric]),
                time=run_info["time"], cpu_time=run_info["cpu_time"],
                starttime=run_info["starttime"], endtime=run_info["endtime"],
                status=StatusType(run_info["status"]),
                additional_info=run_info.get("additional_info") or {},
            ))

        self.evaluate_requested(
            configurations, model=model, splits=splits, metrics=metrics,
            primary_metric=primary_metric, seed=seed, collector=collector,
            cancel_event=cancel_event, told=tell_requested)

        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            # The prior goes in before the ask, so the first model-chosen trial
            # is weighted, and the count comes from SMAC's runhistory, where
            # `add_prior` takes its anchor.
            if prior_report is None and len(smac.runhistory) >= design_end:
                prior_report = _when(self._apply_priors(smac, scenario, config_space, priors),
                                     smac)
            try:
                info = smac.ask()
            except ConfigurationSpaceExhaustedException:
                # A small discrete space can be exhausted, which completes the
                # search.
                collector.stopped_by = STOPPED_BY_EXHAUSTED
                break
            config = dict(info.config)
            all_scores, run_info = evaluate_trial(model, config, splits, metrics, seed=seed,
                                                   keep=self.keep_model is not None)
            cost = to_cost(collector.metric, all_scores[primary_metric])
            # SMAC's runhistory, copied by `serialize_result`, gets the measured
            # timing and the real status, so the surrogate learns from failures.
            smac.tell(info, TrialValue(
                cost=cost, time=run_info["time"], cpu_time=run_info["cpu_time"],
                starttime=run_info["starttime"], endtime=run_info["endtime"],
                status=StatusType(run_info["status"]),
                # The reason tells a bad configuration from a broken model.
                additional_info=run_info.get("additional_info") or {},
            ))
            collector.record(config, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=info.config.origin or "")

            # SMAC trains inside `ask`, so the model is one trial stale, which
            # suffices for a stopping heuristic.
            if wants_confidence and len(collector.results) >= _CONFIDENCE_MIN_TRIALS:
                collector.note_confidence(self._confidence_nothing_better(
                    smac, config_space, collector.incumbent_config))

        incumbent = smac.intensifier.get_incumbent()
        all_trials = (previous_result.trials if previous_result else []) + collector.results

        games = compute_hp_games(config_space, all_trials, metrics, seed=seed,
                                 cancel_event=cancel_event, budget=self.analytics_max_coalitions,
                                 wanted=self.analytics_wanted)

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=dict(incumbent) if incumbent else (all_trials[-1].config if all_trials else {}),
            # `scores` holds the rebased metric, with `score` as the fallback
            # for an old one-metric file.
            best_score=collector.metric.best(
                (t.scores.get(primary_metric, t.score) for t in all_trials), default=0.0),
            **game_fields(games),
            # `metadata` stays in memory, and `experiments/services/run.py`
            # writes the prior event, since `core/` knows nothing of a Run.
            metadata={"smac_output_dir": str(output_dir),
                      "stopped_by": collector.stopped_by,
                      "prior": prior_report},
        )
