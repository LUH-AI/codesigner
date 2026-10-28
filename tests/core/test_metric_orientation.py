"""A metric declares its direction, worst score and cost conversion. The
original four metrics must reproduce higher-is-better, 0.0 and `1 - score`, or
stored `.ihpo` costs read back wrong.
"""

import pytest

from core.metrics import METRICS, Metric, to_cost
from core.optimizers.base import (
    OptimizationResult, TrialCollector, TrialResult, rebase_history)
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS

RMSE = Metric(name="rmse", fn=lambda *_: 0.0, higher_is_better=False, bounds=(0.0, None))


def _trial(n, score, metric="accuracy", status=STATUS_SUCCESS, incumbent=None):
    return TrialResult(
        trial=n, config={"a": n}, scores={metric: score}, score=score,
        incumbent_score=score if incumbent is None else incumbent,
        incumbent_config={"a": n},
        run_info={"status": status, "time": 0.1}, origin="")


def test_a_crash_does_not_become_the_incumbent():
    """With failures scoring 0.0 and the incumbent seeded at -inf, a crash would
    pass `0.0 > -inf` and become the best."""
    collector = TrialCollector(stopping={"max_trials": 10, "max_failures": 5})

    collector.record({"a": 1}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED})

    assert collector.incumbent_config is None
    assert collector.incumbent_score == float("-inf"), "still nothing to beat"


def test_a_crash_cannot_beat_a_real_trial_under_a_lower_is_better_metric():
    """A failure's RMSE is a real number, and a bad real trial can score worse."""
    collector = TrialCollector(stopping={"max_trials": 10, "max_failures": 5},
                               metric=RMSE)
    collector.record({"a": 1}, 900.0, {"rmse": 900.0}, run_info={"status": STATUS_SUCCESS})
    collector.record({"a": 2}, 12.0, {"rmse": 12.0}, run_info={"status": STATUS_CRASHED})

    assert collector.incumbent_config == {"a": 1}
    assert collector.incumbent_score == 900.0


def test_nothing_to_beat_yet_is_the_metric_s_own_worst_end():
    assert TrialCollector(stopping={"max_trials": 1}).incumbent_score == float("-inf")
    assert TrialCollector(stopping={"max_trials": 1},
                          metric=RMSE).incumbent_score == float("inf")


def test_a_lower_is_better_run_improves_downwards():
    collector = TrialCollector(stopping={"max_trials": 10}, metric=RMSE)
    for value in (900.0, 400.0, 700.0, 120.0):
        collector.record({"a": value}, value, {"rmse": value},
                         run_info={"status": STATUS_SUCCESS})

    assert collector.incumbent_score == 120.0


def test_best_index_and_the_incumbent_curve_follow_the_metric():
    scores = [0.4, 0.9, 0.6]
    higher = OptimizationResult(
        trials=[_trial(i + 1, s) for i, s in enumerate(scores)],
        primary_metric="accuracy", best_config={}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={})
    lower = OptimizationResult(
        trials=[_trial(i + 1, s, metric="rmse") for i, s in enumerate(scores)],
        primary_metric="rmse", best_config={}, best_score=0.4,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        declared_metrics={"rmse": RMSE})

    assert higher.best_index("accuracy") == 1        # largest
    assert lower.best_index("rmse") == 0             # smallest
    assert higher.incumbent_scores("accuracy") == [0.4, 0.9, 0.9]
    assert lower.incumbent_scores("rmse") == [0.4, 0.4, 0.4]


def test_rebasing_onto_a_lower_is_better_metric_recomputes_downwards():
    """The rebase also flips the direction of best-so-far."""
    trials = [
        TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.9, "rmse": 800.0},
                    score=0.9, incumbent_score=0.9, incumbent_config={"a": 1},
                    run_info={"status": STATUS_SUCCESS}, origin=""),
        TrialResult(trial=2, config={"a": 2}, scores={"accuracy": 0.5, "rmse": 100.0},
                    score=0.5, incumbent_score=0.9, incumbent_config={"a": 1},
                    run_info={"status": STATUS_SUCCESS}, origin=""),
    ]
    before = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 1}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        declared_metrics={"rmse": RMSE})

    after, stale = rebase_history(before, "rmse")

    assert stale is True
    assert after is not None
    assert after.best_score == 100.0 and after.best_config == {"a": 2}
    assert [t.incumbent_score for t in after.trials] == [800.0, 100.0]


def test_the_stored_cost_is_unchanged_for_every_original_metric(iris_splits, metrics):
    """Old `.ihpo` files carry `1 - score`, and every stored file depends on
    this."""
    from core.optimizers import RandomOptimizer
    from core.models import RandomForestModel

    X_train, X_val, y_train, y_val = iris_splits
    result = RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0)

    stored = RandomOptimizer().serialize_result(result)
    for entry, trial in zip(stored["data"], result.trials):
        assert entry["cost"] == pytest.approx(1.0 - trial.score)


def test_a_cost_metric_is_stored_without_conversion():
    """An imported SMAC objective is stored as it came, since both sides
    minimise it."""
    cost = Metric(name="cost", fn=lambda *_: 0.0,
                  higher_is_better=False, bounds=(None, None))
    result = OptimizationResult(
        trials=[_trial(1, 42.7, metric="cost"), _trial(2, 3.5, metric="cost")],
        primary_metric="cost", best_config={}, best_score=3.5,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        declared_metrics={"cost": cost})

    from core.optimizers import RandomOptimizer
    stored = RandomOptimizer().serialize_result(result)

    assert [e["cost"] for e in stored["data"]] == [42.7, 3.5]
    assert stored["declared_metrics"]["cost"]["higher_is_better"] is False


def test_a_declared_metric_survives_the_round_trip():
    from core.optimizers import RandomOptimizer

    cost = Metric(name="cost", fn=lambda *_: 0.0,
                  higher_is_better=False, bounds=(None, None))
    result = OptimizationResult(
        trials=[_trial(1, 42.7, metric="cost")],
        primary_metric="cost", best_config={}, best_score=42.7,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        declared_metrics={"cost": cost})

    optimizer = RandomOptimizer()
    back = optimizer.deserialize_result(optimizer.serialize_result(result))

    assert back.metric("cost").higher_is_better is False
    assert back.trials[0].score == pytest.approx(42.7), "read back as it was stored"


def test_the_registry_s_own_metrics_are_not_frozen_into_files():
    """Files must leave registry metric definitions to the code."""
    from core.optimizers import RandomOptimizer

    result = OptimizationResult(
        trials=[_trial(1, 0.9)], primary_metric="accuracy",
        best_config={}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        declared_metrics={"accuracy": METRICS["accuracy"]})

    stored = RandomOptimizer().serialize_result(result)
    assert stored["declared_metrics"] == {}


def test_both_writers_agree_on_the_cost_they_store(iris_splits, metrics):
    """`SMACOptimizer.serialize_result` copies SMAC's cost while the base class
    derives it, and only the comparison below ties the two."""
    from core.models import RandomForestModel
    from core.optimizers import RandomOptimizer, SMACOptimizer

    X_train, X_val, y_train, y_val = iris_splits
    result = RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0)

    base = RandomOptimizer().serialize_result(result)
    # `serialize_result` takes the pass-through branch without a live SMAC
    # directory, as on the detail page.
    smac = SMACOptimizer().serialize_result(result)

    assert [e["cost"] for e in base["data"]] == [e["cost"] for e in smac["data"]]


def test_a_failure_still_scores_zero_on_every_original_metric(iris_splits, metrics):
    """Stored files assume a failure scores zero."""
    from core.optimizers.trial import null_scores

    X_train, X_val, y_train, y_val = iris_splits
    from core.splits import holdout

    assert null_scores(metrics, holdout(X_train, y_train, X_val, y_val)) == {
        name: 0.0 for name in metrics}


def test_a_failure_scores_the_null_predictor_not_the_best_one(iris_splits):
    """On an RMSE a crash scoring 0.0 would be the best result."""
    import numpy as np

    from core.optimizers.trial import null_scores
    from core.splits import holdout

    X_train, X_val, y_train, y_val = iris_splits
    rmse = Metric(
        name="rmse",
        fn=lambda y, yp: 0.0,
        higher_is_better=False, bounds=(0.0, None),
        null_score=lambda y: float(np.std(np.arange(len(y), dtype=float))),
    )

    scores = null_scores({"rmse": rmse}, holdout(X_train, y_train, X_val, y_val))

    assert scores["rmse"] > 0.0, "a crash is not a perfect regression"


def test_a_null_score_that_cannot_be_computed_does_not_kill_the_run(iris_splits):
    """Raising on the failure path would kill the run."""
    from core.optimizers.trial import null_scores
    from core.splits import holdout

    X_train, X_val, y_train, y_val = iris_splits
    broken = Metric(name="broken", fn=lambda *_: 0.0,
                    null_score=lambda y: 1 / 0)

    assert null_scores({"broken": broken},
                        holdout(X_train, y_train, X_val, y_val)) == {"broken": 0.0}


def test_no_best_yet_is_stored_as_null_rather_than_negative_infinity():
    """`json.dumps` writes a -inf incumbent as `-Infinity`, which strict parsers
    reject."""
    import json

    from core.optimizers import RandomOptimizer

    collector = TrialCollector(stopping={"max_trials": 10, "max_failures": 50})
    collector.record({"a": 1}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED, "time": 0.1})
    partial = OptimizationResult(
        trials=collector.results, primary_metric="accuracy",
        best_config=collector.incumbent_config or {},
        best_score=collector.incumbent_score,
        hyperparameter_importance={}, hyperparameter_importance_warning={})

    stored = RandomOptimizer().serialize_result(partial)

    assert stored["best_score"] is None
    json.dumps(stored, allow_nan=False)   # raises on infinity


def test_a_stored_null_best_score_resumes_as_nothing_to_beat():
    """The collector already reads None as nothing to beat."""
    from core.optimizers import RandomOptimizer

    partial = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={},
        best_score=float("-inf"),
        hyperparameter_importance={}, hyperparameter_importance_warning={})
    optimizer = RandomOptimizer()
    back = optimizer.deserialize_result(optimizer.serialize_result(partial))

    assert back.best_score is None
    assert TrialCollector(stopping={"max_trials": 1},
                          initial_best_score=back.best_score).incumbent_score == float("-inf")
