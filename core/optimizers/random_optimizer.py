from .base import BaseOptimizer, OptimizationResult, TrialCollector

_MAX_CONSECUTIVE_DUPES = 200  # give up after this many consecutive duplicate samples


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
        n_trials: int,
        previous_result=None,
        seed: int = 0,
        cancel_event=None,
    ) -> OptimizationResult:
        config_space = model.get_config_space(seed=seed)

        evaluated: set = set()
        trial_offset = 0
        if previous_result is not None:
            evaluated = {tuple(sorted(t.config.items())) for t in previous_result.trials}
            trial_offset = len(previous_result.trials)

        collector = TrialCollector(
            target_new_trials=n_trials,
            trial_offset=trial_offset,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
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
            all_scores = model.train_evaluate(
                cfg, X_train, y_train, X_val, y_val, metrics, seed=seed
            )
            collector.record(cfg, all_scores[primary_metric], all_scores)

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
        )


# Module-level sentinel — the loader looks for this name.
OPTIMIZER = RandomOptimizer()
