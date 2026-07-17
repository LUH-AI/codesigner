from core.optimizers import TrialCollector


def test_done_after_target_new_trials():
    """done flips to True exactly when the requested number of new trials is reached.

    Setup: a collector targeting 2 new trials.
    Expect: done is False at 0 and 1 recorded trials, True at 2.
    """
    c = TrialCollector(target_new_trials=2)
    assert not c.done
    c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    assert not c.done
    c.record({"a": 2}, 0.6, {"accuracy": 0.6})
    assert c.done


def test_incumbent_tracks_running_best():
    """Each recorded trial carries the best score/config seen so far (the incumbent).

    Action: record scores 0.5, 0.3, 0.9 in that order.
    Expect: the 0.3 trial still reports the 0.5 incumbent (a worse trial
    never moves it); the 0.9 trial takes over as the new incumbent.
    """
    c = TrialCollector(target_new_trials=3)
    t1 = c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    t2 = c.record({"a": 2}, 0.3, {"accuracy": 0.3})
    t3 = c.record({"a": 3}, 0.9, {"accuracy": 0.9})

    assert t1.incumbent_score == 0.5 and t1.incumbent_config == {"a": 1}
    assert t2.incumbent_score == 0.5 and t2.incumbent_config == {"a": 1}
    assert t3.incumbent_score == 0.9 and t3.incumbent_config == {"a": 3}


def test_trial_offset_produces_sequential_numbers():
    """Trial numbers continue the global sequence when resuming a run.

    Setup: a collector told that 5 trials already exist (trial_offset=5).
    Action: record two new trials.
    Expect: they are numbered 6 and 7 — numbering never restarts at 1.
    """
    c = TrialCollector(target_new_trials=2, trial_offset=5)
    t1 = c.record({"a": 1}, 0.1, {"accuracy": 0.1})
    t2 = c.record({"a": 2}, 0.2, {"accuracy": 0.2})
    assert (t1.trial, t2.trial) == (6, 7)


def test_initial_best_survives_worse_trials():
    """A resumed run's incumbent stays correct relative to the full history.

    Setup: a collector seeded with a previous run's best (0.95, config a=42).
    Action: record a new trial scoring only 0.5.
    Expect: the incumbent still reports the historical best, not the new trial.
    """
    c = TrialCollector(
        target_new_trials=1,
        initial_best_score=0.95,
        initial_best_config={"a": 42},
    )
    t = c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    assert t.incumbent_score == 0.95
    assert t.incumbent_config == {"a": 42}
