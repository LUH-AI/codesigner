"""`sweep_stale_runs` fails runs with status "running" and leaves pending runs
queued in huey.
"""

from contextlib import contextmanager
from datetime import timedelta

import pytest

from experiments.services.run import sweep_stale_runs


@contextmanager
def _immediate(value):
    """`patch.object` cannot restore `immediate`, a property without a deleter.
    """
    from huey.contrib.djhuey import HUEY

    was = HUEY.immediate
    HUEY.immediate = value
    try:
        yield
    finally:
        HUEY.immediate = was


def _experiment():
    from experiments.models import Experiment
    return Experiment.objects.create(
        name="sweep-exp", model_name="Random Forest", optimizer_name="Random Search",
        optimizer_params={}, metric_names=["accuracy"], seed=0,
    )


@pytest.mark.django_db
def test_sweep_marks_running_errored_and_leaves_pending_and_finished():
    from experiments.models import Run

    exp = _experiment()
    running = Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")
    pending = Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="pending")
    done = Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="done")

    swept = sweep_stale_runs()

    running.refresh_from_db()
    pending.refresh_from_db()
    done.refresh_from_db()
    assert swept == 1
    assert running.status == "error" and "restart" in running.error.lower()
    assert pending.status == "pending"  # still queued
    assert done.status == "done"


# `runserver` never calls the startup sweep, and each file save restarts it and
# kills the immediate-mode thread polling the Cancel flag. The run then sits at
# "running" for ever.

@pytest.mark.django_db
def test_a_run_older_than_this_process_had_nobody_executing_it(settings):
    """In immediate mode the web process executes runs, so a run started before
    its boot has no executor; `sweep_orphaned_runs` needs no timeout."""
    from experiments.models import Run
    from experiments.services import run as run_service

    exp = _experiment()
    orphan = Run.objects.create(
        experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
        status="running", started_at=run_service.PROCESS_STARTED - timedelta(minutes=1))

    with _immediate(True):
        run_service._SWEPT = False
        assert run_service.sweep_orphaned_runs() == 1

    orphan.refresh_from_db()
    assert orphan.status == "error"
    assert orphan.finished_at is not None, "or the page waits on it for ever"
    assert "restarts the development server" in orphan.error


@pytest.mark.django_db
def test_a_run_started_since_this_process_booted_is_left_alone(settings):
    from experiments.models import Run
    from experiments.services import run as run_service

    exp = _experiment()
    live = Run.objects.create(
        experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
        status="running", started_at=run_service.PROCESS_STARTED + timedelta(seconds=1))

    with _immediate(True):
        run_service._SWEPT = False
        run_service.sweep_orphaned_runs()

    live.refresh_from_db()
    assert live.status == "running"


@pytest.mark.django_db
def test_with_a_consumer_it_sweeps_nothing(settings):
    """A huey consumer outlives web restarts, so process age says nothing about
    a run; the startup sweep covers consumer restarts."""
    from experiments.models import Run
    from experiments.services import run as run_service

    exp = _experiment()
    Run.objects.create(
        experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
        status="running", started_at=run_service.PROCESS_STARTED - timedelta(hours=4))

    with _immediate(False):
        run_service._SWEPT = False
        assert run_service.sweep_orphaned_runs() == 0


@pytest.mark.django_db
def test_it_runs_once_per_process():
    """Runs started before boot stay orphaned for the process's lifetime, so a
    second sweep finds nothing."""
    from experiments.services import run as run_service

    with _immediate(True):
        run_service._SWEPT = False
        run_service.sweep_orphaned_runs()
        assert run_service.sweep_orphaned_runs() == 0
