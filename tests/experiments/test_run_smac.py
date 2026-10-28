"""Real SMAC runs via the background engine (slow).
"""

import pytest

from tests.conftest import DATASETS_DIR


@pytest.mark.slow
@pytest.mark.django_db
def test_smac_run_completes_and_stores_result():
    from experiments.services import snapshot as adapter
    from experiments.services.run import create_run, execute_run

    snapshot = {
        "version": "0.1.0", "name": "smac-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC (BlackBox)", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    exp = adapter.experiment_from_snapshot(snapshot, adopt_paths=True)
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert exp.data.result is not None
    assert run.status == "done"
    assert len(exp.data.result["data"]) == 3
    assert exp.data.result["optimizer_state"]  # SMAC's working dir


@pytest.mark.slow
@pytest.mark.django_db
def test_smac_resume_through_the_web_engine_keeps_all_trials():
    """`execute_run` keeps the first run's trials on resume. The SMAC Scenario
    names its directory after a facade-metadata hash that differs between
    `BlackBoxFacade` constructions, so a resume could open the wrong directory.
    """
    from experiments.services import snapshot as adapter
    from experiments.services.run import create_run, execute_run

    snapshot = {
        "version": "0.1.0", "name": "smac-resume-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC (BlackBox)", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    exp = adapter.experiment_from_snapshot(snapshot, adopt_paths=True)

    run1 = create_run(exp, {"max_trials": 3}, "accuracy")
    execute_run(run1.id)
    exp.refresh_from_db()
    assert exp.data.result is not None
    assert [e["config_id"] for e in exp.data.result["data"]] == [1, 2, 3]

    run2 = create_run(exp, {"max_trials": 2}, "accuracy")
    execute_run(run2.id)
    run2.refresh_from_db()
    exp.refresh_from_db()
    assert exp.data.result is not None

    assert run2.status == "done"
    assert [e["config_id"] for e in exp.data.result["data"]] == [1, 2, 3, 4, 5]
