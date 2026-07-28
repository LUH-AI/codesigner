"""Step 9: NewExperimentForm gains a gated custom-model upload.

The form accepts *either* a registry model (as before) *or* an uploaded model
.py file, when the ALLOW_CUSTOM_MODELS feature flag is on. An uploaded file is
validated by actually loading it (core.io.load_model_from_path); a valid file
resolves the experiment's model_name to the model's own .name. With the flag
off, the upload field is absent and only registry models are offered.
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
