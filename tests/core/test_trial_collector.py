from core.optimizers import TrialCollector


def test_done_after_the_trial_cap():
    c = TrialCollector(stopping={"max_trials": 2})
    assert not c.done
    c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    assert not c.done
    c.record({"a": 2}, 0.6, {"accuracy": 0.6})
    assert c.done


def test_incumbent_tracks_running_best():
    """`TrialCollector.record` stamps each trial with the incumbent, which a
    worse trial never moves.
    """
    c = TrialCollector(stopping={"max_trials": 3})
    t1 = c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    t2 = c.record({"a": 2}, 0.3, {"accuracy": 0.3})
    t3 = c.record({"a": 3}, 0.9, {"accuracy": 0.9})

    assert t1 is not None and t2 is not None and t3 is not None
    assert t1.incumbent_score == 0.5 and t1.incumbent_config == {"a": 1}
    assert t2.incumbent_score == 0.5 and t2.incumbent_config == {"a": 1}
    assert t3.incumbent_score == 0.9 and t3.incumbent_config == {"a": 3}


def test_trial_offset_produces_sequential_numbers():
    """`trial_offset=5` makes new trials continue at 6 and 7.
    """
    c = TrialCollector(stopping={"max_trials": 2}, trial_offset=5)
    t1 = c.record({"a": 1}, 0.1, {"accuracy": 0.1})
    t2 = c.record({"a": 2}, 0.2, {"accuracy": 0.2})
    assert t1 is not None and t2 is not None
    assert (t1.trial, t2.trial) == (6, 7)


def test_initial_best_survives_worse_trials():
    """`initial_best_score` stays the incumbent when a resumed run scores worse.
    """
    c = TrialCollector(
        stopping={"max_trials": 1},
        initial_best_score=0.95,
        initial_best_config={"a": 42},
    )
    t = c.record({"a": 1}, 0.5, {"accuracy": 0.5})
    assert t is not None
    assert t.incumbent_score == 0.95
    assert t.incumbent_config == {"a": 42}
