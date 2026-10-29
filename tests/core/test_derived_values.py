"""`OptimizationResult` computes `best_index` and `incumbent_scores` once, and
one implementation keeps the chart highlight and the panel on the same trial.
"""

from dataclasses import fields, replace

from core.optimizers.base import OptimizationResult, TrialResult, rebase_history


def _result(scores_by_trial, primary="accuracy"):
    """`_result` builds a result from per-metric scores."""
    trials = []
    for i, scores in enumerate(scores_by_trial, start=1):
        trials.append(TrialResult(
            trial=i, config={"a": i}, scores=dict(scores),
            score=scores[primary], incumbent_score=scores[primary],
            incumbent_config={"a": i}))
    return OptimizationResult(
        trials=trials, primary_metric=primary,
        best_config=trials[-1].config if trials else {},
        best_score=max((s[primary] for s in scores_by_trial), default=0.0),
        hyperparameter_importance={}, hyperparameter_importance_warning={})


def test_best_index_picks_the_highest_score():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.9}, {"accuracy": 0.6}])
    assert r.best_index("accuracy") == 1


def test_best_index_breaks_ties_toward_the_lowest_index():
    """`best_index` breaks ties toward the earliest trial, matching the six
    `max` calls it replaced."""
    r = _result([{"accuracy": 0.5}, {"accuracy": 0.9}, {"accuracy": 0.9}])
    assert r.best_index("accuracy") == 1


def test_best_index_is_per_metric_not_per_result():
    """The page switches metrics client-side against one result."""
    r = _result([{"accuracy": 0.9, "f1": 0.1}, {"accuracy": 0.1, "f1": 0.9}])
    assert r.best_index("accuracy") == 0
    assert r.best_index("f1") == 1


def test_best_index_is_none_with_no_trials():
    """`best_index` gives None for a run cancelled before its first trial."""
    assert _result([]).best_index("accuracy") is None


def test_incumbent_scores_are_the_running_best():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.2}, {"accuracy": 0.7},
                 {"accuracy": 0.5}])
    assert r.incumbent_scores("accuracy") == [0.4, 0.4, 0.7, 0.7]


def test_repeated_calls_reuse_one_computation():
    """Four `performance_over_time` views share one list."""
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.7}])
    assert r.incumbent_scores("accuracy") is r.incumbent_scores("accuracy")


def test_best_index_skips_a_trial_with_no_score_for_the_metric():
    """`io.parse` rejects such files, but rows stored earlier would crash the
    detail page."""
    r = _result([{"accuracy": 0.4, "f1": 0.9}, {"accuracy": 0.9}])
    assert r.best_index("f1") == 0, "the only trial that has an f1"
    assert r.best_index("accuracy") == 1, "unaffected"


def test_best_index_is_none_when_no_trial_has_the_metric():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.9}])
    assert r.best_index("f1") is None


def test_incumbent_scores_carries_the_running_best_across_a_missing_score():
    """`incumbent_scores` keeps one entry per trial position for callers."""
    r = _result([{"accuracy": 0.4, "f1": 0.5}, {"accuracy": 0.9},
                 {"accuracy": 0.2, "f1": 0.8}])
    assert r.incumbent_scores("f1") == [0.5, 0.5, 0.8]


def test_incumbent_scores_leads_with_none_until_the_first_real_score():
    """A floor would draw a value nobody measured."""
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.9, "f1": 0.6}])
    assert r.incumbent_scores("f1") == [None, 0.6]


def test_has_every_score_is_what_the_figure_builders_ask():
    """Per-metric builders return None for partial pictures."""
    r = _result([{"accuracy": 0.4, "f1": 0.5}, {"accuracy": 0.9}])
    assert r.has_every_score("accuracy")
    assert not r.has_every_score("f1")


def test_the_memo_is_not_a_dataclass_field():
    """`rebase_history` uses `dataclasses.replace`, which would carry a memo
    field into the rebased copy.
    """
    names = {f.name for f in fields(OptimizationResult)}
    assert "_derived_memo" not in names
    assert not any(name.startswith("_") for name in names)


def test_a_replaced_result_recomputes_rather_than_inheriting_a_memo():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.9}])
    r.best_index("accuracy")
    fresh = replace(r, trials=list(reversed(r.trials)))
    assert fresh.best_index("accuracy") == 0      # best now first
    assert r.best_index("accuracy") == 1


def test_rebasing_to_another_metric_reports_that_metrics_best():
    """After a metric change, derived values describe the new objective."""
    r = _result([{"accuracy": 0.9, "f1": 0.1}, {"accuracy": 0.1, "f1": 0.9}])
    r.best_index("accuracy")                      # fills accuracy memo

    rebased, stale = rebase_history(r, "f1")
    assert stale
    assert rebased is not None
    assert rebased.best_index("f1") == 1
    assert rebased.incumbent_scores("f1") == [0.1, 0.9]
