"""What ends a run, when it is not simply the trial count.

`TrialCollector.done` is the condition every optimizer loop is written against,
so it is the one place a stopping rule has to be expressed to apply to all of
them. Several criteria can be set at once and the first to fire wins; the
collector records which, because "it stopped at 12 of 30 trials" is only useful
alongside the reason.

The trial cap is not optional. A target score that is never reached, or a
stagnation window on a search that keeps inching forward, would otherwise run
until something else killed it.
"""

import time

import pytest

from core.optimizers.base import STOPPED_BY_TRIALS, TrialCollector


def _record(collector, score, seconds=0.0, n=1):
    """Record *n* trials, all scoring *score* and each taking *seconds*."""
    for _ in range(n):
        collector.record({"x": score}, score, {"accuracy": score},
                         run_info={"time": seconds})


# ── the default: nothing but the trial count ─────────────────────────────────

def test_without_criteria_a_run_ends_at_the_trial_count():
    """The behaviour every existing run has, and must keep."""
    collector = TrialCollector(target_new_trials=3)

    _record(collector, 0.5, n=2)
    assert collector.done is False

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == STOPPED_BY_TRIALS


def test_criteria_that_are_none_are_ignored_rather_than_zero():
    """A form field left blank arrives as None. Read as a limit of zero it
    would stop the run before its first trial."""
    collector = TrialCollector(
        target_new_trials=2,
        stopping={"max_seconds": None, "target_score": None,
                  "max_trial_seconds": None, "no_improvement_trials": None})

    _record(collector, 0.5)
    assert collector.done is False


def test_an_unknown_key_is_not_silently_honoured():
    """Only the declared criteria have meaning; anything else is a typo, and a
    typo must not become a stopping rule nobody wrote."""
    collector = TrialCollector(target_new_trials=2, stopping={"max_trials": 1})

    _record(collector, 0.5)
    assert collector.done is False


# ── each criterion on its own ────────────────────────────────────────────────

def test_the_target_score_ends_the_run():
    collector = TrialCollector(target_new_trials=100, stopping={"target_score": 0.9})

    _record(collector, 0.5)
    assert collector.done is False

    _record(collector, 0.95)
    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_the_target_score_counts_the_incumbent_not_the_last_trial():
    """A trial that scores worse after the target was reached does not un-reach
    it — the incumbent is what the run achieved."""
    collector = TrialCollector(target_new_trials=100, stopping={"target_score": 0.9})

    _record(collector, 0.95)
    _record(collector, 0.1)

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_a_resumed_run_can_already_be_at_its_target():
    """The incumbent carried in from earlier runs counts. Asking for 0.9 when
    0.95 is already in hand should not spend another trial to discover it."""
    collector = TrialCollector(target_new_trials=10, initial_best_score=0.95,
                               stopping={"target_score": 0.9})

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_the_compute_budget_counts_time_spent_in_trials():
    """Distinct from wall clock: this is the compute actually consumed, which
    is what a shared machine is being asked to budget."""
    collector = TrialCollector(target_new_trials=100,
                               stopping={"max_trial_seconds": 5.0})

    _record(collector, 0.5, seconds=2.0, n=2)
    assert collector.done is False

    _record(collector, 0.5, seconds=2.0)
    assert collector.done is True
    assert collector.stopped_by == "max_trial_seconds"


def test_the_wall_clock_limit_ends_the_run():
    collector = TrialCollector(target_new_trials=100, stopping={"max_seconds": 0.05})

    assert collector.done is False
    time.sleep(0.06)

    assert collector.done is True
    assert collector.stopped_by == "max_seconds"


def test_stagnation_ends_the_run():
    collector = TrialCollector(target_new_trials=100,
                               stopping={"no_improvement_trials": 3})

    _record(collector, 0.5)          # the first trial improves on -inf
    _record(collector, 0.4, n=2)
    assert collector.done is False

    _record(collector, 0.4)
    assert collector.done is True
    assert collector.stopped_by == "no_improvement_trials"


def test_an_improvement_resets_the_stagnation_count():
    collector = TrialCollector(target_new_trials=100,
                               stopping={"no_improvement_trials": 3})

    _record(collector, 0.5)
    _record(collector, 0.4, n=2)
    _record(collector, 0.9)          # better: the window starts again
    _record(collector, 0.8, n=2)

    assert collector.done is False


# ── several at once ──────────────────────────────────────────────────────────

def test_the_first_criterion_to_fire_is_the_one_that_stops_it():
    """Six trials would satisfy the cap; the target is reached at two."""
    collector = TrialCollector(target_new_trials=6,
                               stopping={"target_score": 0.9,
                                         "no_improvement_trials": 10})

    _record(collector, 0.5)
    _record(collector, 0.95)

    assert collector.done is True
    assert collector.stopped_by == "target_score"
    assert len(collector.results) == 2


def test_the_trial_cap_still_applies_when_a_target_is_never_reached():
    """The reason the cap is not optional."""
    collector = TrialCollector(target_new_trials=3, stopping={"target_score": 0.99})

    _record(collector, 0.5, n=3)

    assert collector.done is True
    assert collector.stopped_by == STOPPED_BY_TRIALS


def test_the_reason_is_latched_rather_than_recomputed():
    """`done` is read once per loop iteration and the wall clock keeps moving,
    so a run that finished its trials must not be relabelled a timeout on the
    next read."""
    collector = TrialCollector(target_new_trials=1, stopping={"max_seconds": 0.05})

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == STOPPED_BY_TRIALS

    time.sleep(0.06)
    assert collector.done is True
    assert collector.stopped_by == STOPPED_BY_TRIALS


# ── through a real optimizer ─────────────────────────────────────────────────

def test_an_optimizer_stops_early_and_reports_why(optimizers, models, metrics, iris_splits):
    """The collector is not consulted anywhere else, so this is what proves the
    criteria reach an actual search."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=50, seed=0,
        stopping={"target_score": 0.5})

    assert len(result.trials) < 50
    assert result.metadata["stopped_by"] == "target_score"


def test_an_optimizer_given_no_criteria_behaves_as_before(
        optimizers, models, metrics, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=4, seed=0)

    assert len(result.trials) == 4
    assert result.metadata["stopped_by"] == STOPPED_BY_TRIALS


@pytest.mark.slow
def test_smac_honours_a_stopping_criterion(optimizers, models, metrics, iris_splits):
    """SMAC's loop is the one that also has to keep its ask/tell exchange
    balanced when it ends early."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["SMAC"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=20, seed=0,
        stopping={"target_score": 0.5})

    assert len(result.trials) < 20
    assert result.metadata["stopped_by"] == "target_score"
