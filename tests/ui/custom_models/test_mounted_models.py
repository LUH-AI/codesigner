"""`NewExperimentForm` offers the `.py` files in `mounted_models/` when
ALLOW_CUSTOM_MODELS is on and the directory has any.
"""

import pytest
from django.urls import reverse

from core import datasets
from ui.forms import NewExperimentForm
from experiments.models import Experiment

from tests.ui.custom_models.conftest import VALID_MODEL_SRC
from tests.conftest import post_new_experiment


@pytest.fixture
def mounted_dir(tmp_path, monkeypatch):
    """`mounted_dir` patches `_MODELS_DIR` to hold one valid model."""
    d = tmp_path / "mounted_models"
    d.mkdir()
    (d / "mymodel.py").write_text(VALID_MODEL_SRC, encoding="utf-8")
    monkeypatch.setattr(datasets, "_MODELS_DIR", d)
    return d


def _demo_dataset_path():
    """`_demo_dataset_path` names iris since the models here only classify.
    """
    return datasets.demo_datasets()["iris"]


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
    monkeypatch.setattr(datasets, "_MODELS_DIR", tmp_path / "empty")
    assert "mounted_model" not in NewExperimentForm().fields


def test_selecting_a_mounted_model_resolves_its_name(mounted_dir):
    form = NewExperimentForm({
        "name": "mm", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "task": "classification", "seed": "0",
        "mounted_model": _mounted_path(mounted_dir),
    })
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "My Custom Model"


def test_create_view_adopts_a_mounted_model(client, mounted_dir):
    """The created experiment holds its own copy of the mounted model in MEDIA."""
    from pathlib import Path

    resp = post_new_experiment(client, {
        "name": "mm-create", "model_name": "", "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "task": "classification", "seed": "0",
        "mounted_model": _mounted_path(mounted_dir),
    })
    assert resp.status_code == 302
    exp = Experiment.objects.get(data__name="mm-create")
    assert exp.data.model_name == "My Custom Model"
    assert exp.data.model_file and Path(exp.data.model_file.path).is_file()
