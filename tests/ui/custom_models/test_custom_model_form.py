"""`NewExperimentForm` takes a registry model or, with ALLOW_CUSTOM_MODELS on,
an uploaded `.py` that `core.model_source` reads without running.
"""

import pytest

from core import datasets
from ui.forms import NewExperimentForm


def _demo_dataset_path():
    """`_demo_dataset_path` names iris since the models here only classify.
    """
    return datasets.demo_datasets()["iris"]


def _data(**overrides):
    base = {
        "name": "cm",
        "model_name": "",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(), "task": "classification",
        "seed": "0",
    }
    base.update(overrides)
    return base


def test_upload_field_present_when_flag_on(settings):
    settings.ALLOW_CUSTOM_MODELS = True
    assert "model_file" in NewExperimentForm().fields


def test_upload_field_absent_when_flag_off(settings):
    settings.ALLOW_CUSTOM_MODELS = False
    assert "model_file" not in NewExperimentForm().fields


def test_valid_upload_resolves_model_name(model_upload):
    """`model_name` comes from the upload's declared `name`."""
    form = NewExperimentForm(_data(), {"model_file": model_upload})
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "My Custom Model"


def test_invalid_upload_is_rejected(bad_model_upload):
    """A bad upload yields a form error instead of a 500."""
    form = NewExperimentForm(_data(), {"model_file": bad_model_upload})
    assert not form.is_valid()
    assert "model_file" in form.errors or "__all__" in form.errors


def test_registry_model_still_works_without_upload():
    form = NewExperimentForm(_data(model_name="Random Forest"), {})
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "Random Forest"


def test_requires_a_model_one_way_or_another():
    form = NewExperimentForm(_data(), {})
    assert not form.is_valid()


def test_upload_ignored_when_flag_off(settings, model_upload):
    """Without the upload field, a custom-only submission lacks a model."""
    settings.ALLOW_CUSTOM_MODELS = False
    form = NewExperimentForm(_data(), {"model_file": model_upload})
    assert not form.is_valid()


# uploads are read, never run

def test_a_valid_upload_is_not_executed(tmp_path):
    """`NewExperimentForm` must leave the sentinel file missing, since importing
    an upload would run its code in the web process.
    """
    from django.core.files.uploadedfile import SimpleUploadedFile

    sentinel = tmp_path / "i-ran.txt"
    source = f'''
# /// script
# dependencies = []
# ///
import pathlib
pathlib.Path({str(sentinel)!r}).write_text("executed")

class Sneaky(BaseModel):
    name = "Sneaky"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []
'''
    upload = SimpleUploadedFile("sneaky.py", source.encode(), content_type="text/x-python")

    form = NewExperimentForm(_data(), {"model_file": upload})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "Sneaky"
    assert not sentinel.exists(), "the uploaded file was executed during form validation"


def test_a_file_that_would_crash_on_import_is_still_accepted(tmp_path):
    """The model's own environment settles imports later; the form only checks
    for a model class."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    source = '''
# /// script
# dependencies = ["a-package-that-does-not-exist"]
# ///
import a_package_that_does_not_exist

class Later(BaseModel):
    name = "Later"
    def get_config_space(self, seed=0): return None
    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []
'''
    upload = SimpleUploadedFile("later.py", source.encode(), content_type="text/x-python")

    form = NewExperimentForm(_data(), {"model_file": upload})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "Later"
