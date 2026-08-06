"""Step 6 regression: the detail page renders while a run is active.

The live-status partial is *included* on the detail page (not only served by
the status endpoint), and it reverses URLs from `experiment.pk` — so the detail
context must expose the experiment. This GETs the detail page with an active
run, the exact case a NoReverseMatch slipped through because no earlier test
rendered detail while running.
"""

import pytest
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


@pytest.mark.django_db
def test_detail_renders_while_a_run_is_active(client):
    """Detail renders (200) with the status partial and its cancel/poll URLs."""
    from ui.models import Run
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")

    resp = client.get(reverse("ui:experiment_detail", args=[exp.pk]))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Optimizing" in body
    assert reverse("ui:run_cancel", args=[exp.pk]) in body
    assert reverse("ui:run_status", args=[exp.pk]) in body
