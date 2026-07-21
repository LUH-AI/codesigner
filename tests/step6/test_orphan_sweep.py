"""Step 6: sweeping runs orphaned by a server restart.

In-process run threads don't survive a restart, so any run still marked
pending/running afterwards is stale. sweep_stale_runs (called once on startup)
marks them errored; finished runs are untouched.
"""

import pytest

from web.services.run import sweep_stale_runs


def _experiment():
    from web.models import Experiment
    return Experiment.objects.create(
        name="sweep-exp", model_name="Random Forest", optimizer_name="Random Search",
        optimizer_params={}, metric_names=["accuracy"], seed=0,
    )


@pytest.mark.django_db
def test_sweep_marks_active_runs_errored_and_leaves_finished():
    """Pending/running runs become errored with a restart message; done stays done."""
    from web.models import Run

    exp = _experiment()
    running = Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="running")
    pending = Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="pending")
    done = Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="done")

    swept = sweep_stale_runs()

    running.refresh_from_db()
    pending.refresh_from_db()
    done.refresh_from_db()
    assert swept == 2
    assert running.status == "error" and "restart" in running.error.lower()
    assert pending.status == "error"
    assert done.status == "done"
