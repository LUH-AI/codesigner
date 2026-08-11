"""Step 5: .ihpo export (download) and import (upload) through the browser,
plus delete.

Export streams a parseable, Streamlit-loadable .ihpo built by the snapshot
adapter. Import parses an uploaded file (reusing core.io.parse's validation and
messages), creates an experiment, and lands on its detail page. Delete removes
the experiment after a confirmation.
"""

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core import io

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _make(name_source="test2.ihpo"):
    """Create a saved Experiment from a fixture via the adapter."""
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / name_source).read_bytes()))


# ── Export ──────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_export_downloads_parseable_ihpo(client):
    """Exporting an experiment downloads a parseable .ihpo named after it.

    Expect: 200, an attachment named "{name}.ihpo", and a body that io.parse
    accepts (i.e. a file the Streamlit app could load).
    """
    exp = _make()
    resp = client.get(reverse("ui:experiment_export", args=[exp.pk]))

    assert resp.status_code == 200
    disposition = resp["Content-Disposition"]
    assert "attachment" in disposition
    assert f"{exp.name}.ihpo" in disposition
    assert io.parse(resp.getvalue() if hasattr(resp, "getvalue") else resp.content)["name"] == exp.name


@pytest.mark.django_db
def test_export_body_matches_adapter_snapshot(client):
    """The exported bytes are exactly the adapter's snapshot for the row.

    With provenance: export is the one caller that asks for it. The run engine
    and the detail page go through the same function without it, because they
    only need what the experiment *is* and the fingerprint means reading and
    hashing the dataset.
    """
    from ui.services import snapshot as adapter
    exp = _make()
    resp = client.get(reverse("ui:experiment_export", args=[exp.pk]))
    body = resp.getvalue() if hasattr(resp, "getvalue") else resp.content
    assert json.loads(body) == adapter.snapshot_from_experiment(exp, provenance=True)


# ── Import ──────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_import_page_shows_upload_form(client):
    """GET the import page returns an upload form."""
    resp = client.get(reverse("ui:import_experiment"))
    assert resp.status_code == 200
    assert b'type="file"' in resp.content


@pytest.mark.django_db
def test_import_creates_experiment_and_redirects_to_detail(client):
    """Uploading a valid .ihpo creates the experiment and lands on its detail page."""
    from ui.models import Experiment

    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {"file": f})

    exp = Experiment.objects.get()
    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])


@pytest.mark.django_db
def test_import_rejects_invalid_file(client):
    """An unparseable upload re-renders the page with the load error, no row.

    Mirrors the import command: same "Invalid or unreadable experiment file."
    message, validation no weaker than core.io.parse.
    """
    from ui.models import Experiment

    bad = SimpleUploadedFile("bad.ihpo", b"definitely not json {")
    resp = client.post(reverse("ui:import_experiment"), {"file": bad})

    assert resp.status_code == 200
    assert "Invalid or unreadable experiment file." in resp.content.decode()
    assert Experiment.objects.count() == 0


@pytest.mark.django_db
def test_import_allows_duplicate_names(client):
    """Importing the same file twice creates two experiments; they share a name
    but have distinct identifiers (names are no longer unique)."""
    from ui.models import Experiment

    for _ in range(2):
        with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
            resp = client.post(reverse("ui:import_experiment"), {"file": f})

    assert resp.status_code == 302
    assert Experiment.objects.count() == 2
    ids = list(Experiment.objects.values_list("identifier", flat=True))
    assert len(set(ids)) == 2


@pytest.mark.django_db
def test_cross_app_round_trip(client):
    """A Django-exported file re-imports to an equivalent experiment.

    Export an experiment, feed the downloaded bytes back through import under a
    new name, and confirm the identity fields survive — the interop guarantee.
    """
    from ui.models import Experiment
    exp = _make()
    resp = client.get(reverse("ui:experiment_export", args=[exp.pk]))
    body = resp.getvalue() if hasattr(resp, "getvalue") else resp.content

    snapshot = json.loads(body)
    snapshot["name"] = "reimported"
    reups = SimpleUploadedFile("reimported.ihpo", json.dumps(snapshot).encode("utf-8"))
    client.post(reverse("ui:import_experiment"), {"file": reups})

    reimported = Experiment.objects.get(name="reimported")
    assert reimported.optimizer_name == exp.optimizer_name
    assert reimported.metric_names == exp.metric_names
    assert reimported.result == exp.result


# ── Delete ──────────────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_delete_confirmation_page(client):
    """GET the delete URL shows a confirmation naming the experiment."""
    exp = _make()
    resp = client.get(reverse("ui:experiment_delete", args=[exp.pk]))
    assert resp.status_code == 200
    assert exp.name in resp.content.decode()


@pytest.mark.django_db
def test_delete_removes_and_redirects_home(client):
    """POST to the delete URL removes the experiment and redirects home."""
    from ui.models import Experiment
    exp = _make()
    resp = client.post(reverse("ui:experiment_delete", args=[exp.pk]))
    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:home")
    assert not Experiment.objects.filter(pk=exp.pk).exists()
