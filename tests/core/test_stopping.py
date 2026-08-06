"""What ends a run.

`TrialCollector.done` is the condition every optimizer loop is written against,
so it is the one place a stopping rule has to be expressed to apply to all of
them. Several criteria can be set at once and the first to fire wins; the
collector records which, because "it stopped at 12 of 30 trials" is only useful
alongside the reason.

No criterion is privileged — a trial cap is a limit like any other. What is
required is that there be *at least one*, because a run set up with nothing that
could end it has no end.
"""

import time

import pytest

from core.optimizers.base import NoStoppingCriterion, TrialCollector


def _collector(stopping, **kwargs):
    return TrialCollector(stopping=stopping, **kwargs)


def _record(collector, score, seconds=0.0, n=1):
    """Record *n* trials, all scoring *score* and each taking *seconds*."""
    for _ in range(n):
        collector.record({"x": score}, score, {"accuracy": score},
                         run_info={"time": seconds})


# ── at least one, and no favourites ──────────────────────────────────────────

def test_a_run_with_no_criteria_is_refused():
    """Not defaulted to something arbitrary: the caller has to say how this run
    is meant to end, and there is no longer a trial cap standing in for them."""
    with pytest.raises(NoStoppingCriterion):
        TrialCollector()


def test_criteria_that_are_none_do_not_count_as_criteria():
    """A form field left blank arrives as None. Four blanks is still nothing."""
    with pytest.raises(NoStoppingCriterion):
        _collector({"max_seconds": None, "target_score": None,
                    "max_trial_seconds": None, "no_improvement_trials": None})


def test_an_unknown_key_is_not_a_criterion():
    """Only the declared ones have meaning; anything else is a typo, and a typo
    must not become a stopping rule nobody wrote — nor pass for having set one."""
    with pytest.raises(NoStoppingCriterion):
        _collector({"n_trials": 5})          # the old name, and not a criterion


def test_any_single_criterion_is_enough_on_its_own():
    """Including the ones that are not a count. The trial cap is not required."""
    for stopping in ({"max_trials": 3}, {"max_seconds": 60}, {"target_score": 0.9},
                     {"max_trial_seconds": 60}, {"no_improvement_trials": 5},
                     {"incumbent_confidence": 0.95}):
        assert _collector(stopping).done is False, stopping


# ── each criterion on its own ────────────────────────────────────────────────

def test_the_trial_cap_ends_the_run():
    collector = _collector({"max_trials": 3})

    _record(collector, 0.5, n=2)
    assert collector.done is False

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"


def test_the_target_score_ends_the_run():
    collector = _collector({"target_score": 0.9})

    _record(collector, 0.5)
    assert collector.done is False

    _record(collector, 0.95)
    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_the_target_score_counts_the_incumbent_not_the_last_trial():
    """A trial that scores worse after the target was reached does not un-reach
    it — the incumbent is what the run achieved."""
    collector = _collector({"target_score": 0.9})

    _record(collector, 0.95)
    _record(collector, 0.1)

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_a_resumed_run_can_already_be_at_its_target():
    """The incumbent carried in from earlier runs counts. Asking for 0.9 when
    0.95 is already in hand should not spend another trial to discover it."""
    collector = _collector({"target_score": 0.9}, initial_best_score=0.95)

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_the_compute_budget_counts_time_spent_in_trials():
    """Distinct from wall clock: this is the compute actually consumed, which
    is what a shared machine is being asked to budget."""
    collector = _collector({"max_trial_seconds": 5.0})

    _record(collector, 0.5, seconds=2.0, n=2)
    assert collector.done is False

    _record(collector, 0.5, seconds=2.0)
    assert collector.done is True
    assert collector.stopped_by == "max_trial_seconds"


def test_the_wall_clock_limit_ends_the_run():
    collector = _collector({"max_seconds": 0.05})

    assert collector.done is False
    time.sleep(0.06)

    assert collector.done is True
    assert collector.stopped_by == "max_seconds"


def test_stagnation_ends_the_run():
    collector = _collector({"no_improvement_trials": 3})

    _record(collector, 0.5)          # the first trial improves on -inf
    _record(collector, 0.4, n=2)
    assert collector.done is False

    _record(collector, 0.4)
    assert collector.done is True
    assert collector.stopped_by == "no_improvement_trials"


def test_an_improvement_resets_the_stagnation_count():
    collector = _collector({"no_improvement_trials": 3})

    _record(collector, 0.5)
    _record(collector, 0.4, n=2)
    _record(collector, 0.9)          # better: the window starts again
    _record(collector, 0.8, n=2)

    assert collector.done is False


# ── the surrogate's confidence ───────────────────────────────────────────────

def test_confidence_ends_the_run_when_the_optimizer_is_sure_enough():
    collector = _collector({"incumbent_confidence": 0.95})

    collector.note_confidence(0.80)
    _record(collector, 0.5)
    assert collector.done is False

    collector.note_confidence(0.97)
    assert collector.done is True
    assert collector.stopped_by == "incumbent_confidence"


def test_an_optimizer_that_cannot_answer_never_fires_it():
    """Random and grid search fit no model of the objective. Silence has to read
    as "no answer", not as certainty — otherwise the run would end at once."""
    collector = _collector({"incumbent_confidence": 0.95, "max_trials": 3})

    _record(collector, 0.5)
    assert collector.done is False

    collector.note_confidence(None)
    _record(collector, 0.5)
    assert collector.done is False

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"


def test_falling_below_the_threshold_again_keeps_the_run_going():
    """A later trial can widen the surrogate's uncertainty. Until `done` latches,
    the answer is whatever the model currently says."""
    collector = _collector({"incumbent_confidence": 0.95})

    collector.note_confidence(0.96)
    collector.note_confidence(0.5)

    assert collector.done is False


# ── several at once ──────────────────────────────────────────────────────────

def test_the_first_criterion_to_fire_is_the_one_that_stops_it():
    """Six trials would satisfy the cap; the target is reached at two."""
    collector = _collector({"max_trials": 6, "target_score": 0.9,
                            "no_improvement_trials": 10})

    _record(collector, 0.5)
    _record(collector, 0.95)

    assert collector.done is True
    assert collector.stopped_by == "target_score"
    assert len(collector.results) == 2


def test_a_cap_alongside_a_target_that_is_never_reached():
    """Why pairing an unbounded criterion with a bounded one is worth advising."""
    collector = _collector({"max_trials": 3, "target_score": 0.99})

    _record(collector, 0.5, n=3)

    assert collector.done is True
    assert collector.stopped_by == "max_trials"


def test_the_reason_is_latched_rather_than_recomputed():
    """`done` is read once per loop iteration and the wall clock keeps moving,
    so a run that finished its trials must not be relabelled a timeout on the
    next read."""
    collector = _collector({"max_trials": 1, "max_seconds": 0.05})

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"

    time.sleep(0.06)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"


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


def test_n_trials_is_spelling_for_the_trial_cap(optimizers, models, metrics, iris_splits):
    """Kept on `optimize()` because it is how code asks for a search, but it is
    the same criterion under a different name — including in what gets reported."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=4, seed=0)

    assert len(result.trials) == 4
    assert result.metadata["stopped_by"] == "max_trials"


def test_an_explicit_cap_beats_the_shorthand(optimizers, models, metrics, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=50, seed=0,
        stopping={"max_trials": 3})

    assert len(result.trials) == 3


def test_only_an_optimizer_with_a_surrogate_claims_to_support_confidence(optimizers):
    """What the Run form reads to decide whether to offer the criterion."""
    assert optimizers["SMAC"].supports_confidence_stopping is True
    assert optimizers["Random Search"].supports_confidence_stopping is False
    assert optimizers["Grid Search"].supports_confidence_stopping is False


@pytest.mark.slow
def test_smac_answers_with_its_surrogate_and_stops(optimizers, models, metrics, iris_splits):
    """End to end: a real GP, asked about a thousand sampled configurations, ends
    a run that would otherwise have taken 40 trials."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["SMAC"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=40, seed=0,
        stopping={"max_trials": 40, "incumbent_confidence": 0.6})

    assert result.metadata["stopped_by"] == "incumbent_confidence"
    assert len(result.trials) < 40


@pytest.mark.slow
def test_smac_will_not_stop_on_confidence_before_it_has_evidence(
        optimizers, models, metrics, iris_splits):
    """A GP fitted on a handful of points is confident the way a line through
    two points is. Below the minimum, the criterion cannot end a run."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["SMAC"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0,
        stopping={"max_trials": 3, "incumbent_confidence": 0.01})

    assert len(result.trials) == 3
    assert result.metadata["stopped_by"] == "max_trials"
