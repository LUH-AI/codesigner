"""Changing which metric an experiment optimizes, without corrupting the search.

An experiment accumulates trials, and each trial records a score for *every*
metric. So when the optimized metric changes, none of that history has to be
thrown away — it is re-read under the new metric. Two things depend on that
being done rather than skipped:

* **The incumbent trajectory.** Every optimizer seeds its collector from the
  previous run's best. Left alone, that best is the *old* metric's, so the
  "performance of incumbent" figure draws a curve for an objective nobody is
  optimizing any more.
* **SMAC's surrogate.** It is fitted on the costs in its runhistory. Resuming a
  stored run after a metric change would fit one Gaussian process across two
  different cost functions, and every suggestion after that is steered by an
  objective that does not exist. The stored state is discarded and the history
  replayed with recomputed costs instead — so the search keeps what it learned
  without keeping what is wrong.
"""

import pytest

from core.optimizers.base import OptimizationResult, TrialResult, rebase_history


def _trial(n, cfg, accuracy, f1):
    """One recorded trial, with an incumbent trajectory as accuracy would see it."""
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


# The orderings disagree on purpose: accuracy prefers trial 1, f1 prefers 3.
HISTORY = [
    _trial(1, {"x": 1}, accuracy=0.9, f1=0.1),
    _trial(2, {"x": 2}, accuracy=0.5, f1=0.4),
    _trial(3, {"x": 3}, accuracy=0.6, f1=0.8),
]


# ── re-reading the history ───────────────────────────────────────────────────

def test_the_same_metric_is_left_completely_alone():
    """The common case must not pay for the rare one."""
    result = _result(HISTORY)

    rebased, stale = rebase_history(result, "accuracy")

    assert stale is False
    assert rebased is result


def test_a_changed_metric_re_reads_every_trials_score():
    rebased, stale = rebase_history(_result(HISTORY), "f1")

    assert stale is True
    assert [t.score for t in rebased.trials] == [0.1, 0.4, 0.8]


def test_the_incumbent_trajectory_is_recomputed_not_carried_over():
    """Under accuracy the incumbent is trial 1 throughout. Under f1 it has to
    climb 0.1 → 0.4 → 0.8, which is a different curve entirely."""
    rebased, _ = rebase_history(_result(HISTORY), "f1")

    assert [t.incumbent_score for t in rebased.trials] == [0.1, 0.4, 0.8]
    assert [t.incumbent_config["x"] for t in rebased.trials] == [1, 2, 3]


def test_the_best_follows_the_new_metric():
    rebased, _ = rebase_history(_result(HISTORY), "f1")

    assert rebased.best_score == 0.8
    assert rebased.best_config == {"x": 3}
    assert rebased.primary_metric == "f1"


def test_the_trials_themselves_are_not_rewritten():
    """Only the reading changes. The same configurations were evaluated on the
    same data, and every metric's score stays exactly as it was measured."""
    rebased, _ = rebase_history(_result(HISTORY), "f1")

    assert [t.config for t in rebased.trials] == [t.config for t in HISTORY]
    assert [t.scores for t in rebased.trials] == [t.scores for t in HISTORY]


def test_the_original_result_is_not_mutated():
    """It is the caller's, and the detail page may still be rendering from it."""
    result = _result(HISTORY)

    rebase_history(result, "f1")

    assert result.primary_metric == "accuracy"
    assert result.trials[0].score == 0.9


def test_a_history_predating_multi_metric_scores_is_still_reported_stale():
    """A very old .ihpo recorded only the optimized metric, so there is nothing
    to re-read from. The trials stay as they are — but the optimizer must still
    hear that its state is stale, or it would resume against the wrong
    objective."""
    thin = [TrialResult(trial=1, config={"x": 1}, scores={"accuracy": 0.9},
                        score=0.9, incumbent_score=0.9, incumbent_config={"x": 1})]

    rebased, stale = rebase_history(_result(thin), "f1")

    assert stale is True
    assert rebased.trials[0].score == 0.9


def test_no_previous_result_is_not_a_change():
    assert rebase_history(None, "accuracy") == (None, False)


# ── SMAC: the surrogate must not span two objectives ─────────────────────────

@pytest.fixture
def two_runs(optimizers, models, metrics, iris_splits):
    """A real SMAC experiment run once on accuracy, then resumed on f1."""
    X_train, X_val, y_train, y_val = iris_splits
    smac = optimizers["SMAC"]

    def run(primary, n_trials, previous=None):
        return smac.optimize(
            models["Random Forest"], X_train, y_train, X_val, y_val,
            metrics=metrics, primary_metric=primary, n_trials=n_trials,
            previous_result=previous, seed=0)

    first = run("accuracy", 3)
    return smac, first, run("f1", 2, previous=first)


@pytest.mark.slow
def test_smac_resuming_under_a_new_metric_holds_no_old_costs(two_runs):
    """The regression that matters. Every cost in the rebuilt runhistory must be
    computed from f1; a single 1 - accuracy left in there is a point the
    Gaussian process fits against the wrong objective."""
    smac, _, second = two_runs

    # On iris the two metrics agree to four decimals and coincide outright on a
    # perfect trial, so without this the assertion below could pass while the
    # old costs were still in there.
    assert any(t.scores["accuracy"] != t.scores["f1"] for t in second.trials), \
        "the metrics did not diverge on any trial; this test proves nothing"

    costs = sorted(round(e["cost"], 9) for e in smac.serialize_result(second)["data"])
    expected = sorted(round(1.0 - t.scores["f1"], 9) for t in second.trials)

    assert costs == expected


@pytest.mark.slow
def test_smac_keeps_the_history_across_a_metric_change(two_runs):
    """Discarding the surrogate must not discard the trials — the replay is what
    stops the search restarting blind."""
    _, _, second = two_runs

    assert len(second.trials) == 5
    assert [t.trial for t in second.trials] == [1, 2, 3, 4, 5]
    assert second.best_score == max(t.scores["f1"] for t in second.trials)


@pytest.mark.slow
def test_every_resume_rebuilds_and_replays(optimizers, models, metrics, iris_splits):
    """Resuming used to reuse SMAC's stored directory, which required the
    scenario to be byte-identical between runs — and that is what forced the
    budget to be a constant, which is what stopped the search ever reaching its
    model. A rebuild costs one `tell` per past trial and no evaluations, so the
    history survives and the budget can be honest."""
    X_train, X_val, y_train, y_val = iris_splits
    smac = optimizers["SMAC"]
    first = smac.optimize(models["Random Forest"], X_train, y_train, X_val, y_val,
                          metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0)
    second = smac.optimize(models["Random Forest"], X_train, y_train, X_val, y_val,
                           metrics=metrics, primary_metric="accuracy", n_trials=2,
                           previous_result=first, seed=0)

    assert second.metadata["smac_output_dir"] != first.metadata["smac_output_dir"]
    assert len(second.trials) == 5
    assert [t.trial for t in second.trials] == [1, 2, 3, 4, 5]
