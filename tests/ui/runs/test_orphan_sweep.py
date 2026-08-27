"""Step 6 (updated in Step 10): sweeping runs orphaned by a restart.

Runs now execute through a durable huey queue: a *pending* run is still queued
and will be picked up by the consumer, so it is NOT stale. Only a *running* run
is orphaned — the consumer died mid-execution — so sweep_stale_runs (run once at
consumer/system startup) marks only running runs errored; pending and finished
runs are left untouched.
"""

from contextlib import contextmanager
from datetime import timedelta

import pytest

from ui.services.run import sweep_stale_runs


@contextmanager
def _immediate(value):
    """Whether huey runs a task here or hands it to a consumer.

    Set by hand: `immediate` is a property with no deleter, so `patch.object`
    cannot put it back.
    """
    from huey.contrib.djhuey import HUEY

    was = HUEY.immediate
    HUEY.immediate = value
    try:
        yield
    finally:
        HUEY.immediate = was


def _experiment():
    from ui.models import Experiment
    return Experiment.objects.create(
        name="sweep-exp", model_name="Random Forest", optimizer_name="Random Search",
        optimizer_params={}, metric_names=["accuracy"], seed=0,
    )


@pytest.mark.django_db
def test_sweep_marks_running_errored_and_leaves_pending_and_finished():
    """Only running runs are stale; pending (still queued) and done are untouched."""
    from ui.models import Run

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
    assert pending.status == "pending"  # durably queued — will still be processed
    assert done.status == "done"


# ── the run nobody is executing ──────────────────────────────────────────────
#
# `sweep_stale_runs` above is the startup sweep, which the container entrypoint
# calls and a bare `runserver` never does. That leaves a real hole in
# development, and it was reported as one: two runs stuck at "running" for four
# days, with Cancel doing nothing to them.
#
# Cancelling is a flag the *executing* thread polls. In immediate mode that
# thread lives in the web process (see services/dispatch.py), and the web
# process in development is `runserver`, which restarts on every file save. So
# the thread dies, nobody is left to read the flag, and the run sits at
# "running" for ever with a Cancel button that cannot work.

@pytest.mark.django_db
def test_a_run_older_than_this_process_had_nobody_executing_it(settings):
    """An exact test, not a timeout: in immediate mode the executor *is* this
    process, so a run that started before this process booted has no one running
    it and never will. No guessing about how long a run ought to take."""
    from ui.models import Run
    from ui.services import run as run_service

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
    """It is the one case where something really could be executing it."""
    from ui.models import Run
    from ui.services import run as run_service

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
    """The reasoning does not hold there: the consumer is a different process
    that outlives any number of web restarts, so "older than this process" says
    nothing about whether a run is alive. Consumer restarts are the startup
    sweep's job."""
    from ui.models import Run
    from ui.services import run as run_service

    exp = _experiment()
    Run.objects.create(
        experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
        status="running", started_at=run_service.PROCESS_STARTED - timedelta(hours=4))

    with _immediate(False):
        run_service._SWEPT = False
        assert run_service.sweep_orphaned_runs() == 0


@pytest.mark.django_db
def test_it_runs_once_per_process():
    """It answers a question whose answer cannot change while the process
    lives — nothing started before this process booted is going to start being
    executed later — so asking again is a query for nothing."""
    from ui.services import run as run_service

    with _immediate(True):
        run_service._SWEPT = False
        run_service.sweep_orphaned_runs()
        assert run_service.sweep_orphaned_runs() == 0
