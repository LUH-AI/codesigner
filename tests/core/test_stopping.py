"""`TrialCollector.done` is the stopping check every optimizer loop reads.
`stopped_by` records the first criterion to fire. `.TrialCollector` raises for a run
with no criterion at all.
"""

import time

import pytest

from core.optimizers.base import NoStoppingCriterion, TrialCollector


def _collector(stopping, **kwargs):
    return TrialCollector(stopping=stopping, **kwargs)


def _record(collector, score, seconds=0.0, n=1):
    for _ in range(n):
        collector.record({"x": score}, score, {"accuracy": score},
                         run_info={"time": seconds})


def test_a_run_with_no_criteria_is_refused():
    """`TrialCollector` has no default criterion to fall back on."""
    with pytest.raises(NoStoppingCriterion):
        TrialCollector()


def test_criteria_that_are_none_do_not_count_as_criteria():
    with pytest.raises(NoStoppingCriterion):
        _collector({"max_seconds": None, "target_score": None,
                    "max_trial_seconds": None, "no_improvement_trials": None})


def test_an_unknown_key_is_not_a_criterion():
    """`TrialCollector` ignores undeclared keys, so a typo never counts as a
    criterion."""
    with pytest.raises(NoStoppingCriterion):
        _collector({"n_trials": 5})          # old key name


def test_any_single_criterion_is_enough_on_its_own():
    for stopping in ({"max_trials": 3}, {"max_seconds": 60}, {"target_score": 0.9},
                     {"max_trial_seconds": 60}, {"no_improvement_trials": 5},
                     {"incumbent_confidence": 0.95}):
        assert _collector(stopping).done is False, stopping


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


def test_matching_the_target_exactly_is_not_surpassing_it():
    """`target_score` needs a strictly better incumbent, because the form
    pre-fills the incumbent's score."""
    collector = _collector({"target_score": 0.9})

    _record(collector, 0.9)

    assert collector.done is False


def test_the_target_score_counts_the_incumbent_not_the_last_trial():
    """`done` checks the incumbent, so a later worse trial keeps the target
    reached."""
    collector = _collector({"target_score": 0.9})

    _record(collector, 0.95)
    _record(collector, 0.1)

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_a_resumed_run_can_already_be_at_its_target():
    """`initial_best_score` counts, so a resumed run past its target spends no
    trial."""
    collector = _collector({"target_score": 0.9}, initial_best_score=0.95)

    assert collector.done is True
    assert collector.stopped_by == "target_score"


def test_the_compute_budget_counts_time_spent_in_trials():
    """`max_trial_seconds` budgets compute inside trials, apart from the wall
    clock."""
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

    _record(collector, 0.5)          # first trial sets incumbent
    _record(collector, 0.4, n=2)
    assert collector.done is False

    _record(collector, 0.4)
    assert collector.done is True
    assert collector.stopped_by == "no_improvement_trials"


def test_an_improvement_resets_the_stagnation_count():
    collector = _collector({"no_improvement_trials": 3})

    _record(collector, 0.5)
    _record(collector, 0.4, n=2)
    _record(collector, 0.9)          # better: window restarts
    _record(collector, 0.8, n=2)

    assert collector.done is False


def test_confidence_ends_the_run_when_the_optimizer_is_sure_enough():
    collector = _collector({"incumbent_confidence": 0.95})

    collector.note_confidence(0.80)
    _record(collector, 0.5)
    assert collector.done is False

    collector.note_confidence(0.97)
    assert collector.done is True
    assert collector.stopped_by == "incumbent_confidence"


def test_an_optimizer_that_cannot_answer_never_fires_it():
    """`note_confidence(None)` must never read as certainty, because random and
    grid search have no surrogate."""
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
    """`done` follows the latest confidence until it latches."""
    collector = _collector({"incumbent_confidence": 0.95})

    collector.note_confidence(0.96)
    collector.note_confidence(0.5)

    assert collector.done is False


def test_the_first_criterion_to_fire_is_the_one_that_stops_it():
    collector = _collector({"max_trials": 6, "target_score": 0.9,
                            "no_improvement_trials": 10})

    _record(collector, 0.5)
    _record(collector, 0.95)

    assert collector.done is True
    assert collector.stopped_by == "target_score"
    assert len(collector.results) == 2


def test_a_cap_alongside_a_target_that_is_never_reached():
    collector = _collector({"max_trials": 3, "target_score": 0.99})

    _record(collector, 0.5, n=3)

    assert collector.done is True
    assert collector.stopped_by == "max_trials"


def test_the_reason_is_latched_rather_than_recomputed():
    """`done` latches `stopped_by`, so a finished run never turns into a timeout
    later."""
    collector = _collector({"max_trials": 1, "max_seconds": 0.05})

    _record(collector, 0.5)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"

    time.sleep(0.06)
    assert collector.done is True
    assert collector.stopped_by == "max_trials"


def test_an_optimizer_stops_early_and_reports_why(optimizers, models, metrics, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=50, seed=0,
        stopping={"target_score": 0.5})

    assert len(result.trials) < 50
    assert result.metadata["stopped_by"] == "target_score"


def test_n_trials_is_spelling_for_the_trial_cap(optimizers, models, metrics, iris_splits):
    """`optimize()` maps `n_trials` to `max_trials`, in the report too."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=4, seed=0)

    assert len(result.trials) == 4
    assert result.metadata["stopped_by"] == "max_trials"


def test_an_explicit_cap_beats_the_shorthand(optimizers, models, metrics, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["Random Search"]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=50, seed=0,
        stopping={"max_trials": 3})

    assert len(result.trials) == 3


def test_only_an_optimizer_with_a_surrogate_claims_to_support_confidence(optimizers):
    """The Run form reads `supports_confidence_stopping` before offering the
    option."""
    assert optimizers["SMAC"].supports_confidence_stopping is True
    assert optimizers["Random Search"].supports_confidence_stopping is False
    assert optimizers["Grid Search"].supports_confidence_stopping is False


@pytest.mark.slow
def test_smac_answers_with_its_surrogate_and_stops(optimizers, models, metrics, iris_splits):
    """A GP asked about 1000 sampled configurations ends a 40-trial run early."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["SMAC"]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=40, seed=0,
        stopping={"max_trials": 40, "incumbent_confidence": 0.6})

    assert result.metadata["stopped_by"] == "incumbent_confidence"
    assert len(result.trials) < 40


@pytest.mark.slow
def test_smac_will_not_stop_on_confidence_before_it_has_evidence(
        optimizers, models, metrics, iris_splits):
    """`_CONFIDENCE_MIN_TRIALS` keeps a GP fitted on a few points from ending a
    run."""
    X_train, X_val, y_train, y_val = iris_splits

    result = optimizers["SMAC"]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0,
        stopping={"max_trials": 3, "incumbent_confidence": 0.01})

    assert len(result.trials) == 3
    assert result.metadata["stopped_by"] == "max_trials"


def _fail(collector, times=1):
    for _ in range(times):
        collector.record({"x": 1}, 0.0, {"accuracy": 0.0}, run_info={"status": 2})


def test_the_first_failure_ends_the_run_by_default():
    """`DEFAULT_MAX_FAILURES` is 1, so a model with a fixable bug stops at its
    first failure."""
    collector = _collector({"max_trials": 500})
    _fail(collector)

    assert collector.done is True
    assert collector.stopped_by == "all_failing"


def test_a_run_that_expects_failures_says_so_and_keeps_going():
    """Raising `max_failures` lets a search probe a region known to fail."""
    collector = _collector({"max_trials": 500, "max_failures": 10})
    _fail(collector, 9)

    assert collector.done is False

    _fail(collector)
    assert collector.done is True
    assert collector.stopped_by == "all_failing"


def test_the_total_is_not_reset_by_a_success():
    """`max_failures` counts failures across the whole run."""
    collector = _collector({"max_trials": 500, "max_failures": 3})
    _fail(collector, 2)
    collector.record({"x": 2}, 0.9, {"accuracy": 0.9}, run_info={"status": 1})

    assert collector.done is False, "two of three used"

    _fail(collector)
    assert collector.done is True


def test_a_run_whose_every_trial_fails_gives_up():
    """A model that cannot fit the data fails every trial and would burn the
    budget on zeros. `max_failures` is raised to isolate the consecutive
    limit."""
    from core.optimizers.base import DEFAULT_MAX_CONSECUTIVE_FAILURES

    collector = _collector({"max_trials": 500, "max_failures": 10_000})
    _fail(collector, DEFAULT_MAX_CONSECUTIVE_FAILURES)

    assert collector.done is True
    assert collector.stopped_by == "all_failing"


def test_one_good_trial_resets_the_patience():
    """`max_failures` is raised so only the consecutive streak counts."""
    from core.optimizers.base import DEFAULT_MAX_CONSECUTIVE_FAILURES

    collector = _collector({"max_trials": 500, "max_failures": 10_000})
    _fail(collector, DEFAULT_MAX_CONSECUTIVE_FAILURES - 1)
    collector.record({"x": 2}, 0.9, {"accuracy": 0.9}, run_info={"status": 1})
    _fail(collector, DEFAULT_MAX_CONSECUTIVE_FAILURES - 1)

    assert collector.done is False


def test_the_failure_defaults_are_not_a_stopping_criterion_on_their_own():
    """`TrialCollector` adds failure defaults after its criterion check, so they
    cannot satisfy it."""
    with pytest.raises(NoStoppingCriterion):
        _collector({})


def test_a_trial_with_no_status_counts_as_a_success():
    """Callers may omit run_info, and a missing status must never end the run."""
    collector = _collector({"max_trials": 500})
    for _ in range(30):
        collector.record({"x": 1}, 0.5, {"accuracy": 0.5})

    assert collector.done is False


def test_a_trial_that_was_never_run_is_not_serialized(optimizers, models, metrics,
                                                      iris_splits, monkeypatch):
    """SMAC's runhistory can contain entries the collector never recorded, and
    `serialize_result` must drop them instead of reporting unscored rows."""
    import json

    from pathlib import Path

    X_train, X_val, y_train, y_val = iris_splits
    smac = optimizers["SMAC"]()
    result = smac.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0)

    # Forge an entry SMAC knows and the collector lacks.
    rh_path = next(Path(result.metadata["smac_output_dir"]).rglob("runhistory.json"))
    rh = json.loads(rh_path.read_text())
    rh["data"].append({**rh["data"][0], "config_id": 999, "cost": 1.0})
    rh_path.write_text(json.dumps(rh))

    serialized = smac.serialize_result(result)

    assert [e["config_id"] for e in serialized["data"]] == [1, 2, 3]
    assert all("scores" in e for e in serialized["data"])
    assert serialized["stats"]["finished"] == 3
