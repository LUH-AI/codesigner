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

from ..splits import holdout
from .trial import evaluate_trial

from typing import Optional

from .base import (
    BaseOptimizer, OptimizerParam, OptimizationResult, TrialCollector, merge_stopping, rebase_history,
)

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
    ) -> OptimizationResult:
        # One fold unless the caller divided the data itself; see core.splits.
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)

        # The metric may have changed since the last run; re-read the history
        # under the current one so the incumbent trajectory means what the page
        # says it means. No surrogate here, so nothing else is stale.
        previous_result, _ = rebase_history(previous_result, primary_metric)

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

        collector = TrialCollector(
            trial_offset=len(previous_result.trials) if previous_result else 0,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )

        # Two ways to finish, and the grid's own is not a stopping criterion:
        # running out of configurations is the search being *complete*, not a
        # limit being hit. So the loop ends on either, and only the criteria get
        # to name themselves in `stopped_by`.
        for cfg in to_run:
            if cancel_event and cancel_event.is_set():
                break
            if collector.done:
                break
            all_scores, run_info = evaluate_trial(model, cfg, splits, metrics, seed=seed)
            collector.record(cfg, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=_ORIGIN)

        all_trials = (previous_result.trials if previous_result else []) + collector.results

        games = self.compute_hp_games(config_space, all_trials, metrics, seed=seed,
                                      cancel_event=cancel_event)

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=max(all_trials, key=lambda t: t.scores[primary_metric]).config
                        if all_trials else {},
            best_score=max((t.scores[primary_metric] for t in all_trials), default=0.0),
            hyperparameter_importance=games["tunability"][0],
            hyperparameter_importance_warning=games["tunability"][1],
            hyperparameter_sensitivity=games["sensitivity"][0],
            hyperparameter_sensitivity_warning=games["sensitivity"][1],
            hyperparameter_mistunability=games["mistunability"][0],
            hyperparameter_mistunability_warning=games["mistunability"][1],
            hyperparameter_interactions=games["tunability"][2],
            hyperparameter_interactions_warning=games["tunability"][1],
            trials_limit=grid_size,
            metadata={"stopped_by": collector.stopped_by},
        )

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
