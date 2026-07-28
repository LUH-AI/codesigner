"""Step 6: the background run engine (ui/services/run.py).

`create_run` records a pending run and commits the run's metric onto the
experiment; `execute_run` does the work synchronously (rebuild from the DB, run
the optimizer with a DB-backed cancel flag, write the result and status back);
`DbCancelFlag` reads the cancel request from the database.

The resume and cancel *engine* semantics are already validated against the
Streamlit oracle in tests/core/test_optimizers; here we pin the DB wiring
around them. Real Random Forest + Random Search on iris keeps the runs small
and deterministic.
"""

import pytest

from core import io

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _make_experiment(metric_names=None, primary=None, original=None):
    """A saved experiment with the iris dataset attached and no result yet."""
    from ui.services import snapshot as adapter

    snapshot = {
        "version": "0.1.0",
        "name": "run-exp",
        "model_name": "Random Forest",
        "model_path": "",
        "optimizer_name": "Random Search",
        "optimizer_params": {},
        "primary_metric": primary,
        "original_metric": original,
        "metric_names": metric_names or ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"),
        "result": None,
    }
    return adapter.experiment_from_snapshot(snapshot)


# ── create_run ────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_create_run_records_pending_run_and_commits_first_metric():
    """create_run makes a pending Run and commits the metric on a fresh experiment.

    Expect: Run status "pending", cancel not requested; the experiment's primary
    and original both become the chosen metric (first run pins the original).
    """
    from ui.services.run import create_run

    exp = _make_experiment()
    run = create_run(exp, n_trials=3, optimize_metric="f1")

    assert run.status == "pending"
    assert run.cancel_requested is False
    assert run.n_trials == 3
    assert run.primary_metric == "f1"

    exp.refresh_from_db()
    assert exp.primary_metric == "f1"
    assert exp.original_metric == "f1"


@pytest.mark.django_db
def test_create_run_metric_change_moves_primary_not_original():
    """Optimizing a different metric later moves primary but leaves original."""
    from ui.services.run import create_run

    exp = _make_experiment(primary="accuracy", original="accuracy")
    create_run(exp, n_trials=2, optimize_metric="f1")

    exp.refresh_from_db()
    assert exp.primary_metric == "f1"
    assert exp.original_metric == "accuracy"


# ── execute_run ────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_execute_run_completes_and_stores_result():
    """A run executes to completion, storing the result and marking the run done.

    Expect: status "done" with started_at and finished_at set, and the
    experiment carrying a 3-trial result.
    """
    from ui.services.run import create_run, execute_run

    exp = _make_experiment()
    run = create_run(exp, n_trials=3, optimize_metric="accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done"
    assert run.started_at is not None and run.finished_at is not None
    assert exp.result is not None
    assert len(exp.result["data"]) == 3


@pytest.mark.django_db
def test_execute_run_resumes_from_previous_result():
    """A second run continues the trial history rather than restarting it.

    Expect: after a 3-trial run then a 2-trial run, the stored result holds 5
    trials numbered 1..5 (resume via trial_offset).
    """
    from ui.services.run import create_run, execute_run

    exp = _make_experiment()
    execute_run(create_run(exp, n_trials=3, optimize_metric="accuracy").id)
    execute_run(create_run(exp, n_trials=2, optimize_metric="accuracy").id)

    exp.refresh_from_db()
    trials = exp.result["data"]
    assert [t["config_id"] for t in trials] == [1, 2, 3, 4, 5]


@pytest.mark.django_db
def test_execute_run_honours_a_preset_cancel():
    """A run whose cancel is already requested stops and is marked cancelled.

    Pre-setting cancel_requested makes the DB cancel flag report set on the
    optimizer's first check, so no new trials run — deterministic regardless of
    optimizer speed.
    """
    from ui.models import Run
    from ui.services.run import create_run, execute_run

    exp = _make_experiment()
    run = create_run(exp, n_trials=5, optimize_metric="accuracy")
    Run.objects.filter(pk=run.id).update(cancel_requested=True)

    execute_run(run.id)

    run.refresh_from_db()
    assert run.status == "cancelled"
    assert run.finished_at is not None


@pytest.mark.django_db
def test_execute_run_records_errors():
    """If the run cannot be built or executed, the run is marked errored.

    An experiment naming a metric the app doesn't have fails to rebuild; the
    error is captured on the run rather than crashing the worker.
    """
    from ui.services.run import create_run, execute_run

    exp = _make_experiment(metric_names=["accuracy", "bogus-metric"])
    run = create_run(exp, n_trials=3, optimize_metric="accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    assert run.status == "error"
    assert run.error


# ── DbCancelFlag ─────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_db_cancel_flag_reflects_database():
    """DbCancelFlag.is_set() reports the run's current cancel_requested value."""
    from ui.models import Run
    from ui.services.run import DbCancelFlag

    exp = _make_experiment()
    from ui.services.run import create_run
    run = create_run(exp, n_trials=3, optimize_metric="accuracy")

    flag = DbCancelFlag(run.id, ttl=0)
    assert flag.is_set() is False
    Run.objects.filter(pk=run.id).update(cancel_requested=True)
    assert flag.is_set() is True
