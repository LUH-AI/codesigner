"""The browser exports `.ihpo` files, imports them through `core.io.parse`, and
bins experiments; `tests/ui/test_the_bin.py` covers the bin.
"""

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core import io

from tests.conftest import DATASETS_DIR, FIXTURES_DIR, export_ihpo


def _make(name_source="test2.ihpo"):
    """`_make` saves an Experiment from a fixture."""
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / name_source).read_bytes()))


# Export

@pytest.mark.django_db
def test_export_downloads_parseable_ihpo(client):
    """The download is an attachment named after the experiment.
    """
    exp = _make()
    resp = export_ihpo(client, exp.pk)

    assert resp.status_code == 200
    disposition = resp["Content-Disposition"]
    assert "attachment" in disposition
    assert f"{exp.name}.ihpo" in disposition
    assert io.parse(resp.getvalue() if hasattr(resp, "getvalue") else resp.content)["name"] == exp.name


@pytest.mark.django_db
def test_export_body_matches_adapter_snapshot(client):
    """Export alone asks for provenance, since the fingerprint means hashing the
    dataset.
    """
    from experiments.services import snapshot as adapter
    exp = _make()
    resp = export_ihpo(client, exp.pk)
    body = resp.getvalue() if hasattr(resp, "getvalue") else resp.content
    assert json.loads(body) == adapter.snapshot_from_experiment(exp, provenance=True)


# Import

@pytest.mark.django_db
def test_import_page_shows_upload_form(client):
    resp = client.get(reverse("ui:import_experiment"))
    assert resp.status_code == 200
    assert b'type="file"' in resp.content


@pytest.mark.django_db
def test_import_creates_experiment_and_redirects_to_detail(client):
    from experiments.models import Experiment

    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {"file": f})

    exp = Experiment.objects.get()
    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])


@pytest.mark.django_db
def test_import_rejects_invalid_file(client):
    """The error matches the import command's message.
    """
    from experiments.models import Experiment

    bad = SimpleUploadedFile("bad.ihpo", b"definitely not json {")
    resp = client.post(reverse("ui:import_experiment"), {"file": bad})

    assert resp.status_code == 200
    assert "Invalid or unreadable experiment file." in resp.content.decode()
    assert Experiment.objects.count() == 0


@pytest.mark.django_db
def test_import_allows_duplicate_names(client):
    """The two copies share a name and differ in identifier."""
    from experiments.models import Experiment

    resp = None
    for _ in range(2):
        with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
            resp = client.post(reverse("ui:import_experiment"), {"file": f})

    assert resp is not None
    assert resp.status_code == 302
    assert Experiment.objects.count() == 2
    ids = list(Experiment.objects.values_list("identifier", flat=True))
    assert len(set(ids)) == 2


@pytest.mark.django_db
def test_cross_app_round_trip(client):
    """The identity fields survive an export and re-import.
    """
    from experiments.models import Experiment
    exp = _make()
    resp = export_ihpo(client, exp.pk)
    body = resp.getvalue() if hasattr(resp, "getvalue") else resp.content

    snapshot = json.loads(body)
    snapshot["name"] = "reimported"
    reups = SimpleUploadedFile("reimported.ihpo", json.dumps(snapshot).encode("utf-8"))
    client.post(reverse("ui:import_experiment"), {"file": reups})

    reimported = Experiment.objects.get(data__name="reimported")
    assert reimported.data.optimizer_name == exp.data.optimizer_name
    assert reimported.data.metric_names == exp.data.metric_names
    assert reimported.data.result == exp.data.result


# Delete

@pytest.mark.django_db
def test_delete_confirmation_page(client):
    exp = _make()
    resp = client.get(reverse("ui:experiment_delete", args=[exp.pk]))
    assert resp.status_code == 200
    assert exp.name in resp.content.decode()


@pytest.mark.django_db
def test_delete_bins_and_redirects_home(client):
    """The binned experiment leaves every page, including its own URL."""
    exp = _make()
    resp = client.post(reverse("ui:experiment_delete", args=[exp.pk]))
    exp.refresh_from_db()
    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:home")
    assert exp.deleted_at is not None
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 404
