"""The included status partial reverses URLs from `experiment.pk`, so the detail
context must expose the experiment.
"""

import pytest
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


@pytest.mark.django_db
def test_detail_renders_while_a_run_is_active(client):
    """A running experiment shows cancel and poll URLs."""
    from experiments.models import Run
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")

    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Optimizing" in body
    assert reverse("ui:run_cancel", args=[exp.pk]) in body
    assert reverse("ui:run_status", args=[exp.pk]) in body
