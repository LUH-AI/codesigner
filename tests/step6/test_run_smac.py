"""Step 6: a real SMAC run through the background engine (slow).

Preserves the end-to-end SMAC coverage the old synchronous create+run had:
create an experiment configured for SMAC, execute a small run, and confirm the
result is stored and the run is marked done.
"""

import pytest

from tests.conftest import DATASETS_DIR


@pytest.mark.slow
@pytest.mark.django_db
def test_smac_run_completes_and_stores_result():
    from web.services import snapshot as adapter
    from web.services.run import create_run, execute_run

    snapshot = {
        "version": "0.1.0", "name": "smac-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC (BlackBox)", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    exp = adapter.experiment_from_snapshot(snapshot)
    run = create_run(exp, n_trials=3, optimize_metric="accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done"
    assert len(exp.result["data"]) == 3
    assert exp.result["optimizer_state"]  # SMAC embeds its working dir
