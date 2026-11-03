import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from smac import Scenario
from smac.acquisition.function import PI, IntegratedAcquisitionFunction
from smac.initial_design import AbstractInitialDesign
from smac.model.gaussian_process import GaussianProcess, MCMCGaussianProcess

from ..base import BaseOptimizer
from .history import replay
from .params import (
    EXTENDABLE_DESIGNS, FACADE_DEFAULTS, GP_MODELS, INITIAL_DESIGNS, PARAMS_SCHEMA, RUN_SAFE,
    STRATEGIES, signature_default,
)

# `_UNBOUNDED_BUDGET` replaces a missing trial cap and only sizes the initial
# design.
_UNBOUNDED_BUDGET = 100


def scenario_budget(trial_offset, max_trials=None) -> int:
    """`scenario_budget` gives a run, its re-walk and the figure one budget.
    """
    return int(trial_offset) + int(max_trials or _UNBOUNDED_BUDGET)


class InitialPoints(NamedTuple):
    """`InitialPoints` is a design's size: sampled `points`, `trials` with any
    default configuration, and `bound_by`, the limit that set it.
    """

    points: int
    trials: int
    bound_by: str


def initial_points(n_hyperparameters: int, budget: int, params: Mapping[str, Any]) -> InitialPoints:
    """`initial_points` sizes the initial design for *budget* trials, counting
    those already run. SMAC takes the smaller cap; `initial_points_use_max`
    picks the larger instead. SMAC raises on a design larger than the budget
    less the default configuration.
    """
    def setting(name):
        value = params.get(name)
        if value is None:
            value = next(p.default for p in PARAMS_SCHEMA if p.name == name)
        return value

    use_default = bool(setting("use_default_config"))
    caps = []
    if setting("use_share_cap"):
        share = params.get("share_cap")
        if share is None:
            share = signature_default(AbstractInitialDesign.__init__, "max_ratio")
        caps.append((int(float(share) * budget), f"share cap {share:g} of {budget}"))
    if setting("use_trial_cap"):
        count = params.get("trial_cap")
        if count is None:
            per_hp = signature_default(AbstractInitialDesign.__init__,
                                       "n_configs_per_hyperparameter") or 10
            count = per_hp * n_hyperparameters
            caps.append((int(count), f"trial cap {count} ({per_hp} per hyperparameter)"))
        else:
            caps.append((int(count), f"trial cap {int(count)}"))

    if not caps:
        wanted, bound_by = budget, "both caps off, so the whole budget"
    else:
        pick = max if setting("initial_points_use_max") else min
        wanted, bound_by = pick(caps, key=lambda cap: cap[0])

    room = budget - (1 if use_default else 0)
    if wanted > room:
        wanted, bound_by = room, f"the budget of {budget}"
    points = max(1, wanted)
    return InitialPoints(points, points + (1 if use_default else 0), bound_by)


class FacadeBuilder(BaseOptimizer):
    """`FacadeBuilder` builds the SMAC facade of `.SMACOptimizer`. A blank
    setting takes the selected strategy's default, since `smac.BlackBoxFacade`
    and `smac.HyperparameterOptimizationFacade` define different defaults.
    """

    params_schema = PARAMS_SCHEMA
    run_safe = RUN_SAFE

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
        # `get_params()` returns these for `.ihpo` files.
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
        self._gp_model_type = gp_model_type if gp_model_type in GP_MODELS else "vanilla"
        self._gp_restarts = gp_restarts
        self._gp_normalize_y = bool(gp_normalize_y)

    @classmethod
    def migrate_params(cls, stored: Mapping[str, Any]) -> Mapping[str, Any]:
        """`exploration_trials` becomes the trial cap with the share cap off,
        and `exploration_ratio` becomes the share cap.
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
        """A non-nesting design keeps its recorded size on resume, since
        resizing resamples every point.
        """
        pinned = self._pinned_points(previous_result)
        if pinned is not None:
            return pinned
        return initial_points(len(list(scenario.configspace.values())), scenario.n_trials,
                              self.get_params()).points

    def _pinned_points(self, previous_result) -> int | None:
        """`_pinned_points` returns None unless a non-nesting design resumes
        under the same sampling method.
        """
        if self._initial_design in EXTENDABLE_DESIGNS:
            # A nesting design grows with the budget like an uninterrupted run.
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
        """The form switches strategy client-side, so its placeholders need both
        strategies' defaults.
        """
        return {
            name: {strategy: signature_default(getattr(facade, getter), argument)
                   for strategy, facade in STRATEGIES.items()}
            for name, (getter, argument) in FACADE_DEFAULTS.items()
        }

    def resolved_params(self) -> dict:
        """Blank settings take SMAC's signature defaults; the other strategy's
        settings stay null.
        """
        facade = STRATEGIES[self._search_strategy]
        filled = dict(self.get_params())
        for name, (getter, argument) in FACADE_DEFAULTS.items():
            if filled.get(name) is None:
                filled[name] = signature_default(getattr(facade, getter), argument)
        if filled.get("share_cap") is None and self._use_share_cap:
            filled["share_cap"] = signature_default(
                AbstractInitialDesign.__init__, "max_ratio")
        if filled.get("gp_restarts") is None and self._search_strategy == "gp":
            filled["gp_restarts"] = signature_default(GaussianProcess.__init__, "n_restarts")
        return filled

    @classmethod
    def effective_defaults(cls) -> dict:
        """`effective_defaults` maps strategy to value where defaults differ.
        """
        by_strategy = {s: cls(search_strategy=s).resolved_params() for s in STRATEGIES}
        out = {}
        for param in cls.params_schema:
            if param.name == "search_strategy":
                out[param.name] = param.default
                continue
            values = {s: resolved.get(param.name) for s, resolved in by_strategy.items()}
            if param.depends_on and param.depends_on[0] == "search_strategy":
                values = {param.depends_on[1]: values[param.depends_on[1]]}
            distinct = {repr(v) for v in values.values()}
            out[param.name] = next(iter(values.values())) if len(distinct) == 1 else values
        per_hp = signature_default(AbstractInitialDesign.__init__,
                                   "n_configs_per_hyperparameter") or 10
        out["trial_cap"] = f"{per_hp} × hyperparameters"
        return out

    def _surrogate(self, facade, scenario, log_y=None):
        """The GP facade only exposes `model_type` and `kernel`, so `_surrogate`
        builds the process to set restarts.
        """
        if self._search_strategy == "rf":
            given = {
                "n_trees": self._rf_trees,
                "max_depth": self._rf_max_depth,
                "min_samples_split": self._rf_min_samples_split,
                "min_samples_leaf": self._rf_min_samples_leaf,
                # SMAC zeroes `max_features` above 1.0.
                "ratio_features": (None if self._rf_feature_ratio is None
                                   else min(float(self._rf_feature_ratio), 1.0)),
            }
            given = {k: v for k, v in given.items() if v is not None}
            if log_y is None:
                return facade.get_model(
                    scenario, bootstrapping=self._rf_bootstrapping, **given)
            # `get_model` hardcodes `log_y=True`, and raw costs can be negative.
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
            # The facade's MCMC walker count, rounded even.
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
        return {"use_default_config": self._use_default_config}

    def _facade(self, scenario, target_function, previous_result=None,
                initial_points=None):
        """The facade's `get_*` methods build each component with the strategy's
        defaults.
        """
        facade = STRATEGIES[self._search_strategy]

        design = INITIAL_DESIGNS[self._initial_design]
        # `max_ratio=1.0` and an explicit `n_configs` keep SMAC from re-clamping
        # the count. A rebuild passes the recorded size as *initial_points*.
        initial_design = design(
            scenario, max_ratio=1.0,
            n_configs=(initial_points if initial_points is not None
                       else self._initial_points(scenario, previous_result)))

        acquisition = facade.get_acquisition_function(scenario, xi=self._acquisition_xi)
        if self._acquisition == "pi":
            acquisition = PI(xi=self._acquisition_xi)

        model = self._surrogate(facade, scenario)
        # A plain acquisition function only scores one MCMC member.
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

    @contextmanager
    def _rebuilt(self, config_space, trials, metric_name: str, seed: int, previous_result,
                 n_trials: int, purpose: str):
        """`_rebuilt` yields a facade with *trials* replayed; its target raises
        `RuntimeError`."""
        # The size `_pinned_points` reads.
        recorded = ((getattr(previous_result, "metadata", None) or {})
                    .get("initial_design") or {}).get("n_configs")
        with tempfile.TemporaryDirectory() as directory:
            scenario = Scenario(
                config_space, name="ihpo", n_trials=n_trials, deterministic=True, seed=seed,
                output_directory=Path(directory), **self._scenario_extras())

            def _unreachable(config, seed: int = 0) -> float:
                raise RuntimeError(f"target_function called while {purpose}")

            smac = self._facade(scenario, _unreachable, None, initial_points=recorded)
            replay(smac, config_space, trials, metric_name, seed)
            yield smac, scenario
