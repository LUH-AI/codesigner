"""Step 6 (updated in Step 10): sweeping runs orphaned by a restart.

Runs now execute through a durable huey queue: a *pending* run is still queued
and will be picked up by the consumer, so it is NOT stale. Only a *running* run
is orphaned — the consumer died mid-execution — so sweep_stale_runs (run once at
consumer/system startup) marks only running runs errored; pending and finished
runs are left untouched.
"""

import pytest

from ui.services.run import sweep_stale_runs


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
