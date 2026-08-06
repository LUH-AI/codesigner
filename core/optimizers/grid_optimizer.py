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

from .base import (
    BaseOptimizer, OptimizerParam, OptimizationResult, TrialCollector, rebase_history,
)

_NUMERIC_STEPS = 5


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
        n_trials: int,
        previous_result=None,
        seed: int = 0,
        cancel_event=None,
    ) -> OptimizationResult:
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
            target_new_trials=len(to_run),
            trial_offset=len(previous_result.trials) if previous_result else 0,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
        )

        for cfg in to_run:
            if cancel_event and cancel_event.is_set():
                break
            all_scores, run_info = evaluate_trial(
                model, cfg, X_train, y_train, X_val, y_val, metrics, seed=seed)
            collector.record(cfg, all_scores[primary_metric], all_scores, run_info=run_info)

        all_trials = (previous_result.trials if previous_result else []) + collector.results

        hp_importance: dict = {}
        hp_warning: dict = {}
        for metric_name in metrics:
            imp, warn = self.compute_hp_importance(
                config_space, all_trials, metric_name, seed=seed
            )
            hp_importance[metric_name] = imp
            hp_warning[metric_name] = warn

        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=max(all_trials, key=lambda t: t.scores[primary_metric]).config
                        if all_trials else {},
            best_score=max((t.scores[primary_metric] for t in all_trials), default=0.0),
            hyperparameter_importance=hp_importance,
            hyperparameter_importance_warning=hp_warning,
            trials_limit=grid_size,
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
