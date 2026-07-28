"""SMAC records real timing through TrialValue → runhistory → .ihpo.

SMAC can't self-time in ask/tell, so the loop measures the evaluation and feeds
it to TrialValue; SMAC persists it in runhistory, which the serialize override
copies verbatim. This proves the timing survives into the serialized result for
a real SMAC run (slow).
"""

import pytest


@pytest.mark.slow
def test_smac_run_records_and_serializes_timing(optimizers, models, metrics, iris_splits):
    from core.optimizers.base import RUN_INFO_KEYS

    X_train, X_val, y_train, y_val = iris_splits
    result = optimizers["SMAC"].optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=3, seed=0,
    )
    # In-memory TrialResults carry run_info.
    for t in result.trials:
        assert set(t.run_info) == set(RUN_INFO_KEYS)
        assert t.duration >= 0.0
        assert t.run_info["status"] == 1

    # And SMAC's runhistory (copied verbatim by serialize_result) carries the
    # timing on each data entry.
    d = optimizers["SMAC"].serialize_result(result)
    for entry in d["data"]:
        assert entry["time"] >= 0.0
        assert entry["endtime"] >= entry["starttime"]
        assert entry["status"] == 1
