from typing import Optional

from .base import BaseOptimizer, OptimizationResult
from .trial import evaluate_trial

_MAX_CONSECUTIVE_DUPES = 200  # consecutive duplicates before giving up


#: Each trial's origin, as SMAC records one.
_ORIGIN = "Random sample"


class RandomOptimizer(BaseOptimizer):
    """`RandomOptimizer` extends `BaseOptimizer` with ConfigSpace's uniform
    sampler.
    """

    name = "Random Search"

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
        # Random search has no acquisition function, so `priors` is ignored.
        priors=None,
        configurations=None,
    ) -> OptimizationResult:
        splits, previous_result, collector = self._begin(
            previous_result, primary_metric, n_trials, stopping, splits,
            X_train, y_train, X_val, y_val)
        config_space = model.get_config_space(seed=seed)
        evaluated = ({tuple(sorted(t.config.items())) for t in previous_result.trials}
                     if previous_result is not None else set())

        evaluated |= self.evaluate_requested(
            configurations, model=model, splits=splits, metrics=metrics,
            primary_metric=primary_metric, seed=seed, collector=collector,
            cancel_event=cancel_event)

        consecutive_dupes = 0
        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            cfg = dict(config_space.sample_configuration())
            key = tuple(sorted(cfg.items()))
            if key in evaluated:
                consecutive_dupes += 1
                if consecutive_dupes >= _MAX_CONSECUTIVE_DUPES:
                    break
                continue
            consecutive_dupes = 0
            evaluated.add(key)
            all_scores, run_info = evaluate_trial(model, cfg, splits, metrics, seed=seed,
                                                   keep=self.keep_model is not None)
            collector.record(cfg, all_scores[primary_metric], all_scores,
                             run_info=run_info, origin=_ORIGIN)

        return self._finish(collector, previous_result, config_space, metrics, primary_metric,
                            seed, cancel_event)
