"""Each trial records every metric, and a metric change re-reads the history
under the new metric. The incumbent trajectory follows the new metric, and SMAC
replays the history with recomputed costs instead of resuming its stored state.
Otherwise one surrogate would fit two cost functions.
"""

import pytest

from core.optimizers.base import OptimizationResult, TrialResult, rebase_history


def _trial(n, cfg, accuracy, f1):
    """`_trial` builds one trial with accuracy's incumbent trajectory."""
    return TrialResult(
        trial=n, config=cfg, scores={"accuracy": accuracy, "f1": f1},
        score=accuracy, incumbent_score=accuracy, incumbent_config=cfg,
        run_info={"time": 0.1, "cpu_time": 0.1, "starttime": 0.0, "endtime": 0.1,
                  "status": 1, "seed": 0, "instance": None, "budget": None,
                  "additional_info": {}},
    )


def _result(trials, primary="accuracy"):
    best = max(trials, key=lambda t: t.scores[primary]) if trials else None
    return OptimizationResult(
        trials=trials, primary_metric=primary,
        best_config=best.config if best else {},
        best_score=best.scores[primary] if best else 0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        metadata={"smac_output_dir": "/nonexistent"},
    )


# Accuracy prefers trial 1, and f1 prefers trial 3.
HISTORY = [
    _trial(1, {"x": 1}, accuracy=0.9, f1=0.1),
    _trial(2, {"x": 2}, accuracy=0.5, f1=0.4),
    _trial(3, {"x": 3}, accuracy=0.6, f1=0.8),
]


def test_the_same_metric_is_left_completely_alone():
    """The common case pays nothing."""
    result = _result(HISTORY)

    rebased, stale = rebase_history(result, "accuracy")

    assert stale is False
    assert rebased is result


def test_a_changed_metric_re_reads_every_trials_score():
    rebased, stale = rebase_history(_result(HISTORY), "f1")

    assert stale is True
    assert rebased is not None
    assert [t.score for t in rebased.trials] == [0.1, 0.4, 0.8]


def test_the_incumbent_trajectory_is_recomputed_not_carried_over():
    """Under f1 the incumbent climbs 0.1, 0.4, 0.8."""
    rebased, _ = rebase_history(_result(HISTORY), "f1")
    assert rebased is not None

    assert [t.incumbent_score for t in rebased.trials] == [0.1, 0.4, 0.8]
    assert [t.incumbent_config["x"] for t in rebased.trials] == [1, 2, 3]


def test_the_best_follows_the_new_metric():
    rebased, _ = rebase_history(_result(HISTORY), "f1")
    assert rebased is not None

    assert rebased.best_score == 0.8
    assert rebased.best_config == {"x": 3}
    assert rebased.primary_metric == "f1"


def test_the_trials_themselves_are_not_rewritten():
    """`rebase_history` only changes the reading and keeps every measured score."""
    rebased, _ = rebase_history(_result(HISTORY), "f1")
    assert rebased is not None

    assert [t.config for t in rebased.trials] == [t.config for t in HISTORY]
    assert [t.scores for t in rebased.trials] == [t.scores for t in HISTORY]


def test_the_original_result_is_not_mutated():
    """The detail page may still render from the original."""
    result = _result(HISTORY)

    rebase_history(result, "f1")

    assert result.primary_metric == "accuracy"
    assert result.trials[0].score == 0.9


def test_a_history_predating_multi_metric_scores_is_still_reported_stale():
    """Old files only contain the optimized metric, yet the optimizer must still
    hear that its state is stale."""
    thin = [TrialResult(trial=1, config={"x": 1}, scores={"accuracy": 0.9},
                        score=0.9, incumbent_score=0.9, incumbent_config={"x": 1})]

    rebased, stale = rebase_history(_result(thin), "f1")

    assert stale is True
    assert rebased is not None
    assert rebased.trials[0].score == 0.9


def test_no_previous_result_is_not_a_change():
    assert rebase_history(None, "accuracy") == (None, False)


@pytest.fixture
def two_runs(optimizers, models, metrics, iris_splits):
    """SMAC on accuracy, then resumed on f1."""
    X_train, X_val, y_train, y_val = iris_splits
    smac = optimizers["SMAC"]()

    def run(primary, n_trials, previous=None):
        return smac.optimize(
            models["Random Forest"], X_train, y_train, X_val, y_val,
            metrics=metrics, primary_metric=primary, n_trials=n_trials,
            previous_result=previous, seed=0)

    first = run("accuracy", 3)
    return smac, first, run("f1", 2, previous=first)


@pytest.mark.slow
def test_smac_resuming_under_a_new_metric_holds_no_old_costs(two_runs):
    """One leftover accuracy cost would fit the Gaussian process against the
    wrong objective."""
    smac, _, second = two_runs

    # The metrics nearly coincide on iris, and the guard proves they diverge
    # somewhere.
    assert any(t.scores["accuracy"] != t.scores["f1"] for t in second.trials), \
        "the metrics did not diverge on any trial; this test proves nothing"

    costs = sorted(round(e["cost"], 9) for e in smac.serialize_result(second)["data"])
    expected = sorted(round(1.0 - t.scores["f1"], 9) for t in second.trials)

    assert costs == expected


@pytest.mark.slow
def test_smac_keeps_the_history_across_a_metric_change(two_runs):
    """The replay keeps the search from restarting blind."""
    _, _, second = two_runs

    assert len(second.trials) == 5
    assert [t.trial for t in second.trials] == [1, 2, 3, 4, 5]
    assert second.best_score == max(t.scores["f1"] for t in second.trials)


@pytest.mark.slow
def test_every_resume_rebuilds_and_replays(optimizers, models, metrics, iris_splits):
    """A rebuild costs one `tell` per past trial and no evaluations."""
    X_train, X_val, y_train, y_val = iris_splits
    smac = optimizers["SMAC"]()
    first = smac.optimize(models["Random Forest"], X_train, y_train, X_val, y_val,
                          metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0)
    second = smac.optimize(models["Random Forest"], X_train, y_train, X_val, y_val,
                           metrics=metrics, primary_metric="accuracy", n_trials=2,
                           previous_result=first, seed=0)

    assert second.metadata["smac_output_dir"] != first.metadata["smac_output_dir"]
    assert len(second.trials) == 5
    assert [t.trial for t in second.trials] == [1, 2, 3, 4, 5]
