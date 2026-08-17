"""Values derived from a result's trials, computed once (core/optimizers/base.py).

`best_index` and `incumbent_scores` are plain functions of `trials` that the
detail page used to recompute several times per render — the argmax six times
per metric (panels, the selected-config panel, and once per each of
`performance_over_time`'s four views), the incumbent list once per view.

The duplicated cost was trivial. What these pin is the part that wasn't: one
implementation means the highlight the chart draws and the trial the panel below
it describes cannot disagree, and the memo cannot go stale.
"""

from dataclasses import fields, replace

from core.optimizers.base import OptimizationResult, TrialResult, rebase_history


def _result(scores_by_trial, primary="accuracy"):
    """A result whose trials carry whatever per-metric scores are given."""
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
    """Pinned deliberately. Six separate `max(range(...), key=...)` calls all
    happened to break ties this way; consolidating them must not quietly change
    which trial the page calls best."""
    r = _result([{"accuracy": 0.5}, {"accuracy": 0.9}, {"accuracy": 0.9}])
    assert r.best_index("accuracy") == 1


def test_best_index_is_per_metric_not_per_result():
    """Each metric has its own best trial, and the memo must not confuse them —
    the page switches metrics client-side against one rebuilt result."""
    r = _result([{"accuracy": 0.9, "f1": 0.1}, {"accuracy": 0.1, "f1": 0.9}])
    assert r.best_index("accuracy") == 0
    assert r.best_index("f1") == 1


def test_best_index_is_none_with_no_trials():
    """A result can exist with no trials (a run cancelled before its first one),
    and callers index into `trials` with this."""
    assert _result([]).best_index("accuracy") is None


def test_incumbent_scores_are_the_running_best():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.2}, {"accuracy": 0.7},
                 {"accuracy": 0.5}])
    assert r.incumbent_scores("accuracy") == [0.4, 0.4, 0.7, 0.7]


def test_repeated_calls_reuse_one_computation():
    """The whole point: four `performance_over_time` views asking for the same
    metric get the same object, not four identical lists."""
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.7}])
    assert r.incumbent_scores("accuracy") is r.incumbent_scores("accuracy")


def test_the_memo_is_not_a_dataclass_field():
    """Deliberate, and worth pinning against a future tidy-up.

    `rebase_history` rebuilds a result with `dataclasses.replace`, which passes
    declared fields through to `__init__`. A memo declared as a field would be
    carried into the rebased copy; as a plain attribute it is simply absent and
    recomputed. Nothing observable distinguishes the two *today* — rebasing
    changes `score` but not `scores`, which is what these read — so this test
    exists to keep that from becoming load-bearing by accident.
    """
    names = {f.name for f in fields(OptimizationResult)}
    assert "_derived_memo" not in names
    assert not any(name.startswith("_") for name in names)


def test_a_replaced_result_recomputes_rather_than_inheriting_a_memo():
    r = _result([{"accuracy": 0.4}, {"accuracy": 0.9}])
    r.best_index("accuracy")                      # prime the memo
    fresh = replace(r, trials=list(reversed(r.trials)))
    assert fresh.best_index("accuracy") == 0      # 0.9 is now first
    assert r.best_index("accuracy") == 1          # the original is untouched


def test_rebasing_to_another_metric_reports_that_metrics_best():
    """The end-to-end case `rebase_history` exists for: after a metric change,
    the derived values must describe the new objective."""
    r = _result([{"accuracy": 0.9, "f1": 0.1}, {"accuracy": 0.1, "f1": 0.9}])
    r.best_index("accuracy")                      # prime under the old metric

    rebased, stale = rebase_history(r, "f1")
    assert stale
    assert rebased.best_index("f1") == 1
    assert rebased.incumbent_scores("f1") == [0.1, 0.9]
