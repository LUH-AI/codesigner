"""Step 10: pick a server-side "mounted" model instead of uploading one.

InteractiveHPO's headless fallback offered a dropdown of .py models found in a
mounted directory (core.io.mounted_models(), scanning mounted_models/). This is
the Docker-volume workflow: mount a models/ dir and choose from it without
uploading. It's gated by ALLOW_CUSTOM_MODELS (loading one is arbitrary code
execution, same as an upload) and only appears when the directory has models.

A chosen mounted model is adopted into MEDIA on create (via the snapshot
adapter's existing model_path adoption), so the experiment is self-contained.
"""

import pytest
from django.urls import reverse

from core import io
from ui.forms import NewExperimentForm
from ui.models import Experiment

from tests.step9.conftest import VALID_MODEL_SRC


@pytest.fixture
def mounted_dir(tmp_path, monkeypatch):
    """A mounted_models/ dir holding one valid model .py, wired into core.io."""
    d = tmp_path / "mounted_models"
    d.mkdir()
    (d / "mymodel.py").write_text(VALID_MODEL_SRC, encoding="utf-8")
    monkeypatch.setattr(io, "_MODELS_DIR", d)
    return d


def _demo_dataset_path():
    return next(iter(io.demo_datasets().values()))


def _mounted_path(mounted_dir):
    return str(mounted_dir / "mymodel.py")


def test_form_offers_mounted_models_when_present(mounted_dir, settings):
    settings.ALLOW_CUSTOM_MODELS = True
    field = NewExperimentForm().fields["mounted_model"]
    values = [v for v, _label in field.choices]
    assert _mounted_path(mounted_dir) in values


def test_form_hides_mounted_models_when_flag_off(mounted_dir, settings):
    settings.ALLOW_CUSTOM_MODELS = False
    assert "mounted_model" not in NewExperimentForm().fields


def test_form_hides_mounted_models_when_none_present(tmp_path, monkeypatch, settings):
    settings.ALLOW_CUSTOM_MODELS = True
    monkeypatch.setattr(io, "_MODELS_DIR", tmp_path / "empty")  # nonexistent → {}
    assert "mounted_model" not in NewExperimentForm().fields


def test_selecting_a_mounted_model_resolves_its_name(mounted_dir):
    form = NewExperimentForm({
        "name": "mm", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "seed": "0",
        "mounted_model": _mounted_path(mounted_dir),
    })
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "My Custom Model"


def test_create_view_adopts_a_mounted_model(client, mounted_dir):
    """Creating with a mounted model copies it into MEDIA and stores the
    resolved model name — the experiment is runnable and self-contained."""
    from pathlib import Path

    resp = client.post(reverse("ui:new_experiment"), {
        "name": "mm-create", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "seed": "0",
        "mounted_model": _mounted_path(mounted_dir),
    })
    assert resp.status_code == 302
    exp = Experiment.objects.get(name="mm-create")
    assert exp.model_name == "My Custom Model"
    assert exp.model_file and Path(exp.model_file.path).is_file()
