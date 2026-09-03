"""NewExperimentForm accepts a gated custom-model upload.

The form accepts *either* a registry model (as before) *or* an uploaded model
.py file, when the ALLOW_CUSTOM_MODELS feature flag is on. The upload is
*read*, not run (core.model_source): a valid file resolves the experiment's
model_name from the literal name in its class. With the flag off, the upload
field is absent and only registry models are offered.
"""

import pytest

from core import io
from ui.forms import NewExperimentForm


def _demo_dataset_path():
    """Path of a bundled demo dataset, to satisfy the form's dataset rule."""
    return next(iter(io.demo_datasets().values()))


def _data(**overrides):
    base = {
        "name": "cm",
        "model_name": "",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(),
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
    """A valid uploaded model file makes the form valid and sets model_name to
    the model's declared .name — no registry model needed."""
    form = NewExperimentForm(_data(), {"model_file": model_upload})
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "My Custom Model"


def test_invalid_upload_is_rejected(bad_model_upload):
    """A .py with no BaseModel subclass is rejected with a form error, not a
    500 — the file is loaded during validation."""
    form = NewExperimentForm(_data(), {"model_file": bad_model_upload})
    assert not form.is_valid()
    assert "model_file" in form.errors or "__all__" in form.errors


def test_registry_model_still_works_without_upload():
    """The pre-existing path — pick a registry model, no upload — is unchanged."""
    form = NewExperimentForm(_data(model_name="Random Forest"), {})
    assert form.is_valid(), form.errors
    assert form.cleaned_data["model_name"] == "Random Forest"


def test_requires_a_model_one_way_or_another():
    """Neither a registry model nor an upload → a validation error."""
    form = NewExperimentForm(_data(), {})
    assert not form.is_valid()


def test_upload_ignored_when_flag_off(settings, model_upload):
    """With the flag off there is no upload field, so a custom-only submission
    falls back to requiring a registry model (and fails without one)."""
    settings.ALLOW_CUSTOM_MODELS = False
    form = NewExperimentForm(_data(), {"model_file": model_upload})
    assert not form.is_valid()


# ── the file is read, never run ──────────────────────────────────────────────

def test_a_valid_upload_is_not_executed(tmp_path):
    """The point of reading the source instead of importing it.

    Validating by import meant an upload ran arbitrary code in the web process,
    for as long as its imports took, before anyone had agreed to anything. This
    model writes a file when its module body runs; submitting the form must
    leave that file uncreated.
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
    """Whether a model imports cleanly is settled later, in its own
    environment. The form's job is only to tell whether it *is* a model."""
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
