"""Step 10: runs execute through a huey task queue instead of a raw thread.

The run lifecycle (create → execute → done, resume, cancel, error) is unchanged
and already covered in tests/step6 by calling `execute_run` directly. Here we
pin the queue seam: `start_background_run` now enqueues a huey `db_task` that
calls `execute_run`, and the web process only enqueues (state stays in the DB,
so polling/cancel are untouched). In immediate mode the task runs synchronously,
letting us assert the whole launch → execute path end to end.
"""

import pytest

from tests.conftest import DATASETS_DIR


def _runnable_experiment():
    """A saved experiment with iris attached, ready to run (Random Search)."""
    from ui.services import snapshot as adapter
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
    })


def test_start_background_run_executes_via_the_queue():
    """Enqueuing a run (immediate mode) runs the task and completes the run,
    writing the result back to the experiment — the launch→task→execute path."""
    from ui.services.run import create_run, start_background_run

    exp = _runnable_experiment()
    run = create_run(exp, n_trials=3, optimize_metric="accuracy")
    start_background_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done", run.error
    assert exp.result is not None
    assert len(exp.result["data"]) == 3


def test_run_experiment_task_is_a_registered_huey_task():
    """The run body is a huey db_task (TaskWrapper): it can be enqueued
    (`schedule`) and run directly (`call_local`), and the consumer autoloads it
    from ui/tasks.py."""
    from ui.tasks import run_experiment_task

    assert hasattr(run_experiment_task, "schedule")   # huey TaskWrapper API
    assert hasattr(run_experiment_task, "call_local")  # db_task raw function


def test_call_local_runs_the_experiment():
    """Running the task's wrapped function directly (bypassing the queue)
    executes the run — the consumer's execution path."""
    from ui.services.run import create_run
    from ui.tasks import run_experiment_task

    exp = _runnable_experiment()
    run = create_run(exp, n_trials=2, optimize_metric="accuracy")
    run_experiment_task.call_local(run.id)

    run.refresh_from_db()
    assert run.status == "done", run.error


def test_inline_runs_are_dispatched_off_the_request_thread(settings, monkeypatch):
    """With no consumer running (immediate mode, i.e. a lone `runserver`), the
    work must not happen during the click: `start_background_run` hands it to a
    background thread and returns straight away, so the browser gets its page
    back instead of waiting out the whole optimization.
    """
    import threading

    from ui.services import run as run_service

    settings.RUN_IMMEDIATE_IN_THREAD = True
    started, release = threading.Event(), threading.Event()

    def blocking_task(run_id):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr("ui.tasks.run_experiment_task", blocking_task)
    run_service.start_background_run(1)

    # The call returned while the task is still blocked — i.e. it did not run
    # inline. (Without the thread this line is only reached after `release`.)
    assert started.wait(timeout=5), "task never started"
    assert not release.is_set()
    release.set()


def test_inline_runs_stay_synchronous_when_threading_is_off(settings, monkeypatch):
    """With the dev-server dispatch off (the default under tests and in
    production, where a real consumer runs), enqueuing executes inline so
    callers and assertions see a finished run."""
    from ui.services import run as run_service

    settings.RUN_IMMEDIATE_IN_THREAD = False
    calls = []
    monkeypatch.setattr("ui.tasks.run_experiment_task", lambda run_id: calls.append(run_id))
    run_service.start_background_run(7)

    assert calls == [7]
