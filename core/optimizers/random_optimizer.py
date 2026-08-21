from typing import Optional

from .base import BaseOptimizer, OptimizationResult, TrialCollector, merge_stopping, rebase_history
from ..splits import holdout
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
    params_schema = []

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

        evaluated: set = set()
        trial_offset = 0
        if previous_result is not None:
            evaluated = {tuple(sorted(t.config.items())) for t in previous_result.trials}
            trial_offset = len(previous_result.trials)

        collector = TrialCollector(
            trial_offset=trial_offset,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )

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
            hyperparameter_moebius=games["tunability"][3],
            hyperparameter_sensitivity_interactions=games["sensitivity"][2],
            hyperparameter_sensitivity_moebius=games["sensitivity"][3],
            hyperparameter_mistunability_interactions=games["mistunability"][2],
            hyperparameter_mistunability_moebius=games["mistunability"][3],
            hyperparameter_tunability_total=games["tunability"][4],
            metadata={"stopped_by": collector.stopped_by},
        )
