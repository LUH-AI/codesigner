import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.model_selection import ParameterGrid

from .trial import evaluate_trial

from typing import Optional

from .base import BaseOptimizer, OptimizationResult, OptimizerParam

_NUMERIC_STEPS = 5


#: Where every configuration this optimizer proposes comes from — one point of
#: the grid, in order. See `_ORIGIN` in random_optimizer.
_ORIGIN = "Grid point"


class GridOptimizer(BaseOptimizer):
    """Exhaustive grid search over a discretized hyperparameter space.

    Continuous and large integer ranges are sampled into a fixed number of
    evenly-spaced values.  All valid combinations (respecting ConfigSpace
    conditions) are evaluated in order up to n_trials.  Resume skips configs
    already present in previous_result.
    """

    name = "Grid Search"

    params_schema = [
        OptimizerParam("numeric_steps", "Numeric HP grid steps", "int", default=_NUMERIC_STEPS, min=2, max=50),
    ]

    def __init__(self, numeric_steps: int = _NUMERIC_STEPS):
        self._numeric_steps = numeric_steps

    def optimize(
        self,
        model,
        X_train, y_train,
        X_val, y_val,
        metrics: dict,
        primary_metric: str,
        n_trials: int | None = None,
        previous_result=None,
        seed: int = 0,
        cancel_event=None,
        stopping: Optional[dict] = None,
        splits=None,
        # Accepted and ignored: a search that fits no model of the objective has
        # no acquisition function for a prior to weight. The caller passes what
        # the reader stated without having to know which optimizer will get it.
        priors=None,
        configurations=None,
        projected_priors=None,
    ) -> OptimizationResult:
        splits, previous_result, collector = self._begin(
            previous_result, primary_metric, n_trials, stopping, splits,
            X_train, y_train, X_val, y_val)

        config_space = model.get_config_space(seed=seed)
        hps = list(config_space.values())
        param_grid = {hp.name: self._hp_values(hp) for hp in hps}

        full_grid = [
            cfg for cfg in ParameterGrid(param_grid)
            if self._is_valid(cfg, config_space)
        ]
        grid_size = len(full_grid)

        evaluated: set = set()
        if previous_result is not None:
            evaluated = {tuple(sorted(t.config.items())) for t in previous_result.trials}

        to_run = [
            cfg for cfg in full_grid
            if tuple(sorted(cfg.items())) not in evaluated
        ][:n_trials]

        requested = self.evaluate_requested(
            configurations, model=model, splits=splits, metrics=metrics,
            primary_metric=primary_metric, seed=seed, collector=collector,
            cancel_event=cancel_event)
        to_run = [cfg for cfg in to_run if tuple(sorted(cfg.items())) not in requested]

        # Two ways to finish, and the grid's own is not a stopping criterion:
        # running out of configurations is the search being *complete*, not a
        # limit being hit. So the loop ends on either, and only the criteria get
        # to name themselves in `stopped_by`.
        for cfg in to_run:
            if cancel_event and cancel_event.is_set():
                break
            if collector.done:
                break
            all_scores, run_info = evaluate_trial(model, cfg, splits, metrics, seed=seed,
                                                   keep=self.keep_model is not None)
            collector.record(cfg, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=_ORIGIN)

        return self._finish(collector, previous_result, config_space, metrics, primary_metric,
                            seed, cancel_event, trials_limit=grid_size)


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hp_values(self, hp) -> list:
        """Discrete list of values to evaluate for one hyperparameter."""
        if isinstance(hp, CategoricalHyperparameter):
            return list(hp.choices)
        if isinstance(hp, OrdinalHyperparameter):
            return list(hp.sequence)
        if isinstance(hp, Constant):
            return [hp.value]
        if isinstance(hp, (UniformIntegerHyperparameter, NormalIntegerHyperparameter)):
            n_values = hp.upper - hp.lower + 1
            if n_values <= self._numeric_steps:
                return list(range(hp.lower, hp.upper + 1))
            return [int(round(v)) for v in np.linspace(hp.lower, hp.upper, self._numeric_steps)]
        if isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
            return np.linspace(hp.lower, hp.upper, self._numeric_steps).tolist()
        return [hp.default_value]

    @staticmethod
    def _is_valid(config_dict: dict, config_space) -> bool:
        try:
            Configuration(config_space, values=config_dict)
            return True
        except Exception:
            return False
