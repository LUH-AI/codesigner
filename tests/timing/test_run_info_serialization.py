"""run_info round-trips through the .ihpo serialize/deserialize, uniformly.

A trial's run_info is splatted into its data entry on serialize and collected
back on deserialize, so timing survives a save/load. Trials without run_info
(old files) emit no timing keys and deserialize to duration 0.0 — backward
compatible with existing fixtures.
"""

from core.optimizers.base import RUN_INFO_KEYS, OptimizationResult, TrialResult


def _result(trials):
    return OptimizationResult(
        trials=trials, primary_metric="accuracy",
        best_config=trials[-1].config if trials else {},
        best_score=max((t.score for t in trials), default=0.0),
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )


def _trial(n, run_info=None):
    return TrialResult(trial=n, config={"a": n}, scores={"accuracy": 0.5 + n / 100},
                       score=0.5 + n / 100, incumbent_score=0.5 + n / 100,
                       incumbent_config={"a": n}, run_info=run_info or {})


def test_run_info_survives_round_trip(optimizers):
    opt = optimizers["Random Search"]
    ri = {"instance": None, "seed": 0, "budget": None, "time": 0.5, "cpu_time": 0.4,
          "status": 1, "starttime": 1000.0, "endtime": 1000.5, "additional_info": {}}
    result = _result([_trial(1, ri)])
    back = opt.deserialize_result(opt.serialize_result(result))
    assert back.trials[0].run_info == ri
    assert back.trials[0].duration == 0.5


def test_serialized_entry_carries_all_run_info_keys(optimizers):
    opt = optimizers["Random Search"]
    ri = {k: (1 if k == "status" else (0.0 if k in ("time", "cpu_time", "starttime", "endtime") else None))
          for k in RUN_INFO_KEYS}
    ri["additional_info"] = {}
    d = opt.serialize_result(_result([_trial(1, ri)]))
    entry = d["data"][0]
    for k in RUN_INFO_KEYS:
        assert k in entry


def test_empty_run_info_emits_no_timing_keys(optimizers):
    """A trial with no run_info serializes to an entry without timing keys, and
    deserializes back to an empty run_info (duration 0)."""
    opt = optimizers["Random Search"]
    d = opt.serialize_result(_result([_trial(1)]))
    entry = d["data"][0]
    assert not any(k in entry for k in ("time", "starttime", "endtime", "cpu_time"))
    back = opt.deserialize_result(d)
    assert back.trials[0].run_info == {}
    assert back.trials[0].duration == 0.0
