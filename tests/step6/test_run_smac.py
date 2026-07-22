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


@pytest.mark.slow
@pytest.mark.django_db
def test_smac_resume_through_the_web_engine_keeps_all_trials():
    """Resuming a SMAC run through create_run/execute_run must not lose the
    first run's trials — the exact path a user driving the browser exercises.

    Reproduces a real bug: SMAC's Scenario auto-generates its on-disk
    directory name from a hash of facade metadata that isn't stable across
    separate BlackBoxFacade constructions, so a resumed run (a fresh
    optimizer instance rebuilt from the saved Experiment row, exactly what
    execute_run does every time) could persist its new trials to a
    different directory than the one being read back from — silently
    dropping the accumulated history from the saved result.
    """
    from web.services import snapshot as adapter
    from web.services.run import create_run, execute_run

    snapshot = {
        "version": "0.1.0", "name": "smac-resume-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC (BlackBox)", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    exp = adapter.experiment_from_snapshot(snapshot)

    run1 = create_run(exp, n_trials=3, optimize_metric="accuracy")
    execute_run(run1.id)
    exp.refresh_from_db()
    assert [e["config_id"] for e in exp.result["data"]] == [1, 2, 3]

    run2 = create_run(exp, n_trials=2, optimize_metric="accuracy")
    execute_run(run2.id)
    run2.refresh_from_db()
    exp.refresh_from_db()

    assert run2.status == "done"
    assert [e["config_id"] for e in exp.result["data"]] == [1, 2, 3, 4, 5]
