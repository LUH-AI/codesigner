"""The new-experiment form offers a "Use demo datasets" checkbox and a "Use
demo models" checkbox — mirroring InteractiveHPO — instead of showing both the
demo and upload inputs at once. Both default to unchecked (upload).

The checkboxes are a UI affordance (JS shows/hides and disables the inactive
input); the form's validation is unchanged, so a normal demo+registry
submission still creates an experiment.
"""

from django.urls import reverse

from ui.models import Experiment


def _demo_dataset_path():
    from core import io
    return next(iter(io.demo_datasets().values()))


def test_dataset_toggle_present(client):
    body = client.get(reverse("ui:new_experiment")).content.decode()
    assert 'name="use_demo_dataset"' in body  # the demo/upload checkbox


def test_dataset_checkbox_defaults_unchecked(client):
    """The demo checkbox is off by default, so upload is the initial mode."""
    body = client.get(reverse("ui:new_experiment")).content.decode()
    checkbox = body[body.index('name="use_demo_dataset"') - 60:body.index('name="use_demo_dataset"') + 60]
    assert "checked" not in checkbox


def test_model_toggle_present_when_flag_on(client, settings):
    settings.ALLOW_CUSTOM_MODELS = True
    body = client.get(reverse("ui:new_experiment")).content.decode()
    assert 'name="use_demo_model"' in body


def test_model_toggle_absent_when_flag_off(client, settings):
    settings.ALLOW_CUSTOM_MODELS = False
    body = client.get(reverse("ui:new_experiment")).content.decode()
    assert 'name="use_demo_model"' not in body


def test_normal_submission_still_creates(client):
    """The toggles don't break the ordinary path — demo dataset + registry
    model still creates an experiment."""
    resp = client.post(reverse("ui:new_experiment"), {
        "name": "toggle-ok",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": _demo_dataset_path(),
        "seed": "0",
    })
    assert resp.status_code == 302
    assert Experiment.objects.filter(name="toggle-ok").exists()
