import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from smac import Scenario
from smac.acquisition.function import PI, IntegratedAcquisitionFunction
from smac.initial_design import AbstractInitialDesign
from smac.model.gaussian_process import GaussianProcess, MCMCGaussianProcess

from ..base import BaseOptimizer
from .history import replay
from .params import (
    EXTENDABLE_DESIGNS, FACADE_DEFAULTS, GP_MODELS, INITIAL_DESIGNS, PARAMS_SCHEMA, STRATEGIES,
    signature_default,
)

# `_UNBOUNDED_BUDGET` stands in for a missing trial cap and only sizes the
# initial design, where the share cap is the real setting.
_UNBOUNDED_BUDGET = 100


def scenario_budget(trial_offset, max_trials=None) -> int:
    """`scenario_budget` serves a run, its re-walk and the figure, which
    therefore quote the same decay factor.
    """
    return int(trial_offset) + int(max_trials or _UNBOUNDED_BUDGET)


class FacadeBuilder(BaseOptimizer):
    """`FacadeBuilder` stores the configuration of `.SMACOptimizer` and
    constructs the SMAC facade from it: the initial design, the surrogate
    model, the acquisition function and the acquisition maximizer.
    `.SMACOptimizer` extends `FacadeBuilder` with the optimization loop and
    with the methods that reconstruct the search of a finished run. A blank
    setting resolves to the default of the strategy selected from `.STRATEGIES`,
    since `smac.BlackBoxFacade` and `smac.HyperparameterOptimizationFacade`
    define different defaults.
    """

    params_schema = PARAMS_SCHEMA

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
        self._gp_model_type = gp_model_type if gp_model_type in GP_MODELS else "vanilla"
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
                share = signature_default(AbstractInitialDesign.__init__, "max_ratio")
            caps.append(int(float(share) * budget))
        if self._use_trial_cap:
            count = self._trial_cap
            if count is None:
                # SMAC's per-hyperparameter count, read from its signature.
                per_hp = signature_default(AbstractInitialDesign.__init__,
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
        if self._initial_design in EXTENDABLE_DESIGNS:
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
            name: {strategy: signature_default(getattr(facade, getter), argument)
                   for strategy, facade in STRATEGIES.items()}
            for name, (getter, argument) in FACADE_DEFAULTS.items()
        }

    def resolved_params(self) -> dict:
        """Blank settings take SMAC's signature defaults, and the SMAC version
        sits in `environment.packages.smac`. Settings of the other strategy stay
        null, which re-imports correctly.
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
            replay(smac, config_space, trials, metric_name, seed)
            yield smac, scenario
