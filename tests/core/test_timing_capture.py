"""`timed_evaluation` fills the SMAC runhistory fields of `RUN_INFO_KEYS`, and
`TrialResult.duration` reads their wall time.
"""

import time as _time

import pytest

from core.optimizers.base import RUN_INFO_KEYS, TrialCollector, TrialResult
from core.optimizers.timing import timed_evaluation


def test_timed_evaluation_fills_all_run_info_keys():
    with timed_evaluation(seed=7) as run_info:
        _time.sleep(0.01)
    assert set(run_info) == set(RUN_INFO_KEYS)
    assert run_info["time"] >= 0.0
    assert run_info["cpu_time"] >= 0.0
    assert run_info["endtime"] >= run_info["starttime"] > 0.0
    assert run_info["status"] == 1
    assert run_info["seed"] == 7
    assert run_info["budget"] is None and run_info["instance"] is None
    assert run_info["additional_info"] == {}


def test_trialresult_duration_reads_run_info_time():
    t = TrialResult(trial=1, config={}, scores={"accuracy": 0.5}, score=0.5,
                    incumbent_score=0.5, incumbent_config={}, run_info={"time": 1.25})
    assert t.duration == 1.25


def test_trialresult_duration_defaults_to_zero():
    t = TrialResult(trial=1, config={}, scores={"accuracy": 0.5}, score=0.5,
                    incumbent_score=0.5, incumbent_config={})
    assert t.run_info == {}
    assert t.duration == 0.0


def test_collector_record_stores_run_info():
    c = TrialCollector(stopping={"max_trials": 1})
    t = c.record({"k": 1}, 0.5, {"accuracy": 0.5}, run_info={"time": 0.3, "status": 1})
    assert t is not None
    assert t.run_info["time"] == 0.3


@pytest.mark.parametrize("optimizer_key", ["Random Search", "Grid Search"])
def test_optimizer_records_timing_for_every_trial(optimizer_key, optimizers, models, metrics, iris_splits):
    """Each trial of a real run carries the full `run_info`."""
    X_train, X_val, y_train, y_val = iris_splits
    result = optimizers[optimizer_key]().optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0,
    )
    assert result.trials
    for t in result.trials:
        assert set(t.run_info) == set(RUN_INFO_KEYS)
        assert t.duration >= 0.0
        assert t.run_info["endtime"] >= t.run_info["starttime"] > 0.0
        assert t.run_info["status"] == 1
        assert t.run_info["seed"] == 0
        assert t.run_info["budget"] is None and t.run_info["instance"] is None
