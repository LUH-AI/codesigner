"""`start_background_run` enqueues a huey `db_task` calling `execute_run`, and
the database holds the run state.
"""

import pytest

from tests.conftest import DATASETS_DIR


def _runnable_experiment():
    """`_runnable_experiment` saves a Random Search experiment on iris."""
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot({
        "version": "0.1.0",
        "name": "queue-exp",
        "model_name": "Random Forest",
        "model_path": "",
        "optimizer_name": "Random Search",
        "optimizer_params": {},
        "primary_metric": None,
        "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"),
        "result": None,
    }, adopt_paths=True)


def test_start_background_run_executes_via_the_queue():
    from experiments.services.run import create_run, start_background_run

    exp = _runnable_experiment()
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    start_background_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done", run.error
    assert exp.data.result is not None
    assert len(exp.data.result["data"]) == 3


def test_run_experiment_task_is_a_registered_huey_task():
    """The huey consumer autoloads `run_experiment_task` from
    `experiments/tasks.py`."""
    from experiments.tasks import run_experiment_task

    assert hasattr(run_experiment_task, "schedule")   # TaskWrapper API
    assert hasattr(run_experiment_task, "call_local")  # raw function


def test_call_local_runs_the_experiment():
    """`call_local` runs the consumer's execution path, bypassing the queue."""
    from experiments.services.run import create_run
    from experiments.tasks import run_experiment_task

    exp = _runnable_experiment()
    run = create_run(exp, {"max_trials": 2}, "accuracy")
    run_experiment_task.call_local(run.id)

    run.refresh_from_db()
    assert run.status == "done", run.error


def test_inline_runs_are_dispatched_off_the_request_thread(settings, monkeypatch):
    """In immediate mode, `start_background_run` hands the optimization to a
    background thread and returns at once.
    """
    import threading

    from experiments.services import run as run_service

    settings.RUN_IMMEDIATE_IN_THREAD = True
    started, release = threading.Event(), threading.Event()

    def blocking_task(run_id):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr("experiments.tasks.run_experiment_task", blocking_task)
    run_service.start_background_run(1)

    # The call returned while the task still blocks on `release`.
    assert started.wait(timeout=5), "task never started"
    assert not release.is_set()
    release.set()


def test_inline_runs_stay_synchronous_when_threading_is_off(settings, monkeypatch):
    """Tests switch `RUN_IMMEDIATE_IN_THREAD` off, and enqueuing then runs
    inline so callers see a finished run."""
    from experiments.services import run as run_service

    settings.RUN_IMMEDIATE_IN_THREAD = False
    calls = []
    monkeypatch.setattr("experiments.tasks.run_experiment_task", lambda run_id: calls.append(run_id))
    run_service.start_background_run(7)

    assert calls == [7]
