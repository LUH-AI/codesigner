"""Step 6: the create/run split and the run lifecycle in the browser.

Creating an experiment no longer runs it; the detail page launches a background
run, polls its status, and can cancel it. The metric-change confirmation gates
a run that would change the optimized metric. Background execution is patched
out here (tested directly in test_run_execution) so these stay deterministic.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR


@pytest.fixture
def no_thread(monkeypatch):
    """Stop launched runs from actually executing, so a launched run stays
    'pending' and assertions are race-free."""
    from web.services import run as run_service
    monkeypatch.setattr(run_service, "start_background_run", lambda run_id: None)


def _create(client, **overrides):
    data = {
        "name": "exp",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "seed": 0,
    }
    data.update(overrides)
    return client.post(reverse("web:new_experiment"), data)


def _experiment(**overrides):
    from web.services import snapshot as adapter
    snapshot = {
        "version": "0.1.0", "name": "exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": overrides.get("primary_metric"),
        "original_metric": overrides.get("original_metric"),
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    return adapter.experiment_from_snapshot(snapshot)


# ── Create no longer runs ─────────────────────────────────────────────────────

@pytest.mark.django_db
def test_create_persists_without_running(client):
    """Creating an experiment saves it with no result and launches no run.

    Expect: a redirect to the detail page, an Experiment with result None and no
    committed metric, and zero Run rows.
    """
    from web.models import Experiment, Run

    resp = _create(client, name="fresh")
    exp = Experiment.objects.get(name="fresh")

    assert resp.status_code == 302
    assert resp["Location"] == reverse("web:experiment_detail", args=[exp.pk])
    assert exp.result is None
    assert exp.primary_metric is None
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_detail_offers_a_run_form_when_idle(client):
    """An idle experiment with a dataset shows a Run form (trials + metric)."""
    exp = _experiment()
    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    assert 'name="n_trials"' in html
    assert 'name="optimize_metric"' in html


# ── Launching a run ────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_run_launches_a_pending_run(client, no_thread):
    """Submitting the Run form records a pending run and redirects to detail."""
    from web.models import Run

    exp = _experiment()
    resp = client.post(reverse("web:experiment_run", args=[exp.pk]),
                       {"n_trials": 3, "optimize_metric": "accuracy"})

    assert resp.status_code == 302
    assert resp["Location"] == reverse("web:experiment_detail", args=[exp.pk])
    run = Run.objects.get(experiment=exp)
    assert run.status == "pending"
    assert run.n_trials == 3
    assert run.primary_metric == "accuracy"


@pytest.mark.django_db
def test_run_metric_change_asks_for_confirmation(client, no_thread):
    """Changing the optimized metric first shows a confirmation, launching nothing.

    Expect: 200 (a confirmation naming both metrics), and no run created yet.
    """
    from web.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    resp = client.post(reverse("web:experiment_run", args=[exp.pk]),
                       {"n_trials": 3, "optimize_metric": "f1"})

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "f1" in body and "accuracy" in body
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_run_metric_change_confirm_new_launches_with_chosen(client, no_thread):
    """Confirming 'new' launches a run optimizing the chosen metric."""
    from web.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    resp = client.post(reverse("web:experiment_run", args=[exp.pk]),
                       {"n_trials": 3, "optimize_metric": "f1", "decision": "new"})

    assert resp.status_code == 302
    run = Run.objects.get(experiment=exp)
    assert run.primary_metric == "f1"
    exp.refresh_from_db()
    assert exp.primary_metric == "f1"
    assert exp.original_metric == "accuracy"


@pytest.mark.django_db
def test_run_metric_change_confirm_old_keeps_original(client, no_thread):
    """Confirming 'old' launches a run optimizing the original metric."""
    from web.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    client.post(reverse("web:experiment_run", args=[exp.pk]),
                {"n_trials": 3, "optimize_metric": "f1", "decision": "old"})

    run = Run.objects.get(experiment=exp)
    assert run.primary_metric == "accuracy"


# ── Status polling ──────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_status_partial_reports_running(client):
    """While a run is active, the status endpoint shows it running and keeps polling."""
    from web.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="running")
    resp = client.get(reverse("web:run_status", args=[exp.pk]))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "unning" in body            # "Running" / "running"
    assert "hx-trigger" in body        # keeps polling
    assert reverse("web:run_cancel", args=[exp.pk]) in body  # cancel available


@pytest.mark.django_db
def test_status_partial_refreshes_when_finished(client):
    """When no run is active, the status endpoint asks the page to refresh."""
    from web.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="done")
    resp = client.get(reverse("web:run_status", args=[exp.pk]))

    assert resp.status_code == 200
    assert resp.get("HX-Refresh") == "true"


# ── Cancel ─────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_cancel_requests_cancellation(client):
    """Cancelling sets the active run's cancel flag (the worker stops soon after)."""
    from web.models import Run

    exp = _experiment()
    run = Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="running")
    resp = client.post(reverse("web:run_cancel", args=[exp.pk]))

    run.refresh_from_db()
    assert run.cancel_requested is True
    assert resp.status_code in (302, 200)


@pytest.mark.django_db
def test_delete_works_with_an_active_run(client):
    """Deleting an experiment mid-run removes it (its runs cascade)."""
    from web.models import Experiment, Run

    exp = _experiment()
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="running")
    resp = client.post(reverse("web:experiment_delete", args=[exp.pk]))

    assert resp.status_code == 302
    assert not Experiment.objects.filter(pk=exp.pk).exists()
    assert Run.objects.count() == 0


# ── Sidebar spinner ──────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_sidebar_marks_running_experiments(client):
    """An experiment with an active run is flagged in the sidebar (spinner)."""
    from web.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    Run.objects.create(experiment=exp, n_trials=3, primary_metric="accuracy", status="running")
    html = client.get(reverse("web:home")).content.decode()

    assert "spinner" in html.lower()
