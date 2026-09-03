"""Size/extension limits on uploads — the create form's file fields and the
import view's request.FILES handling, which has no Form to hang a validator
off (see ui/validators.py, shared by both paths)."""

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from ui.forms import NewExperimentForm

from tests.conftest import DATASETS_DIR, FIXTURES_DIR

pytestmark = pytest.mark.django_db


def _valid_post(**overrides):
    data = {
        "name": "my-exp", "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "seed": 0,
    }
    data.update(overrides)
    return data


# ── NewExperimentForm ────────────────────────────────────────────────────────

def test_dataset_upload_wrong_extension_is_rejected():
    upload = SimpleUploadedFile("dataset.txt", b"a,b\n1,2\n")
    form = NewExperimentForm(
        _valid_post(demo_dataset=""), {"dataset_file": upload})

    assert not form.is_valid()
    assert "dataset_file" in form.errors


def test_dataset_upload_too_large_is_rejected(settings):
    settings.MAX_UPLOAD_BYTES = 10
    upload = SimpleUploadedFile("dataset.csv", b"a,b\n1,2\n1,2\n1,2\n")
    form = NewExperimentForm(
        _valid_post(demo_dataset=""), {"dataset_file": upload})

    assert not form.is_valid()
    assert "dataset_file" in form.errors


def test_model_upload_wrong_extension_is_rejected(settings):
    settings.ALLOW_CUSTOM_MODELS = True
    upload = SimpleUploadedFile("model.txt", b"x = 1\n")
    form = NewExperimentForm(_valid_post(model_name=""), {"model_file": upload})

    assert not form.is_valid()
    assert "model_file" in form.errors


def test_model_upload_too_large_is_rejected(settings):
    settings.ALLOW_CUSTOM_MODELS = True
    settings.MAX_UPLOAD_BYTES = 10
    upload = SimpleUploadedFile("model.py", b"x = 1\n" * 10)
    form = NewExperimentForm(_valid_post(model_name=""), {"model_file": upload})

    assert not form.is_valid()
    assert "model_file" in form.errors


def test_a_dataset_upload_within_the_limit_still_works(settings):
    """The limit rejects what's over it and nothing else."""
    settings.MAX_UPLOAD_BYTES = 1024 * 1024
    with open(DATASETS_DIR / "iris.csv", "rb") as f:
        upload = SimpleUploadedFile("iris.csv", f.read())
    form = NewExperimentForm(_valid_post(demo_dataset=""), {"dataset_file": upload})

    assert form.is_valid(), form.errors


# ── import_experiment ────────────────────────────────────────────────────────

def test_import_rejects_an_oversized_ihpo_file(client, settings):
    from ui.models import Experiment

    settings.MAX_UPLOAD_BYTES = 10
    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {"file": f})

    assert resp.status_code == 200
    assert "upload limit" in resp.content.decode()
    assert Experiment.objects.count() == 0


def test_import_rejects_a_wrong_extension_dataset_reupload(client):
    from ui.models import Experiment

    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {
            "file": f, "dataset": SimpleUploadedFile("data.txt", b"a,b\n1,2\n")})

    assert resp.status_code == 200
    assert ".csv" in resp.content.decode()
    assert Experiment.objects.count() == 0


def test_import_rejects_an_oversized_dataset_reupload(client, settings):
    from ui.models import Experiment

    settings.MAX_UPLOAD_BYTES = 10
    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {
            "file": f,
            "dataset": SimpleUploadedFile("wine.csv", (DATASETS_DIR / "wine.csv").read_bytes())})

    assert resp.status_code == 200
    assert "upload limit" in resp.content.decode()
    assert Experiment.objects.count() == 0


def test_import_rejects_a_wrong_extension_model_reupload(client, settings):
    """Only reached when ALLOW_CUSTOM_MODELS is on — the same gate the create
    form's model_file field is hidden behind."""
    from ui.models import Experiment

    settings.ALLOW_CUSTOM_MODELS = True
    with open(FIXTURES_DIR / "test2.ihpo", "rb") as f:
        resp = client.post(reverse("ui:import_experiment"), {
            "file": f, "model": SimpleUploadedFile("model.txt", b"x = 1\n")})

    assert resp.status_code == 200
    assert ".py" in resp.content.decode()
    assert Experiment.objects.count() == 0
