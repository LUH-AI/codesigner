from typing import Optional

from .base import BaseOptimizer, OptimizationResult
from .trial import evaluate_trial

_MAX_CONSECUTIVE_DUPES = 200  # give up after this many consecutive duplicate samples


#: Where every configuration this optimizer proposes comes from. Recorded per
#: trial like SMAC's, so a reader of the file does not have to know which
#: optimizer wrote it to know how a configuration was arrived at.
_ORIGIN = "Random sample"


class RandomOptimizer(BaseOptimizer):
    """Uniform random search over the hyperparameter space.

    Samples configurations independently at random using ConfigSpace's built-in
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
