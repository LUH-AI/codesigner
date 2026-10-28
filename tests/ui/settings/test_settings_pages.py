"""A ⚙ link reaches an experiment's settings, and the Settings tab lands on the
appearance subpage.
"""

from django.urls import reverse

from experiments.models import Experiment, GlobalSettings
from ui.experiment_settings import SETTING_DEFAULTS


def _exp():
    return Experiment.objects.create(
        name="s", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)





def test_experiment_settings_use_default_checkbox_clears_override(client):
    exp = _exp()
    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"use_default_settings": "on", "show_trials": "on"})
    exp.refresh_from_db()
    assert exp.use_default_settings is True
    assert exp.settings == {}


def test_default_experiment_settings_saves_global(client):
    """An empty form writes every key, with omitted checkboxes off."""
    client.post(reverse("ui:default_experiment_settings"), {})
    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["show_trials"] is False
    assert set(stored) == set(SETTING_DEFAULTS)


def test_appearance_page_renders(client):
    resp = client.get(reverse("ui:appearance"))
    assert resp.status_code == 200


def test_appearance_page_links_to_defaults(client):
    body = client.get(reverse("ui:appearance")).content.decode()
    assert reverse("ui:default_experiment_settings") in body


def test_experiment_page_has_settings_link(client):
    exp = _exp()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert reverse("ui:experiment_settings", args=[exp.pk]) in body


def test_sidebar_has_settings_link(client):
    body = client.get(reverse("ui:home")).content.decode()
    assert reverse("ui:appearance") in body
