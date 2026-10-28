"""`timed_evaluation` measures each evaluation, because SMAC cannot time its own
trials under ask and tell.
"""

import pytest


@pytest.mark.slow
def test_smac_run_records_and_serializes_timing(optimizers, models, metrics, iris_splits):
    from core.optimizers.base import RUN_INFO_KEYS

    X_train, X_val, y_train, y_val = iris_splits
    optimizer = optimizers["SMAC"]()
    result = optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0,
    )
    # in memory
    for t in result.trials:
        assert set(t.run_info) == set(RUN_INFO_KEYS)
        assert t.duration >= 0.0
        assert t.run_info["status"] == 1

    # serialized runhistory entries carry timing
    d = optimizer.serialize_result(result)
    for entry in d["data"]:
        assert entry["time"] >= 0.0
        assert entry["endtime"] >= entry["starttime"]
        assert entry["status"] == 1
