"""The settings pages: per-experiment, global, and default-experiment-settings.

Per-experiment settings inherit the global defaults (a checkbox), can override
them, and can be reset to them. The defaults subpage edits the global template.
Both are reachable — a ⚙ link on the experiment page and a global link in the
sidebar.
"""

from django.urls import reverse

from web.models import Experiment, GlobalSettings


def _exp():
    return Experiment.objects.create(
        name="s", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def test_experiment_settings_page_renders(client):
    exp = _exp()
    resp = client.get(reverse("web:experiment_settings", args=[exp.pk]))
    assert resp.status_code == 200
    body = resp.content.decode()
    assert 'name="use_default_settings"' in body
    assert 'name="export_absolute_times"' in body


def test_experiment_settings_saves_an_override(client):
    exp = _exp()
    client.post(reverse("web:experiment_settings", args=[exp.pk]),
                {"export_absolute_times": ""})  # use_default unchecked, export unchecked
    exp.refresh_from_db()
    assert exp.use_default_settings is False
    assert exp.settings == {"export_absolute_times": False}


def test_experiment_settings_reset_restores_defaults(client):
    exp = _exp()
    exp.use_default_settings = False
    exp.settings = {"export_absolute_times": False}
    exp.save()
    client.post(reverse("web:experiment_settings", args=[exp.pk]), {"reset": "1"})
    exp.refresh_from_db()
    assert exp.use_default_settings is True
    assert exp.settings == {}


def test_experiment_settings_use_default_checkbox_clears_override(client):
    exp = _exp()
    client.post(reverse("web:experiment_settings", args=[exp.pk]),
                {"use_default_settings": "on", "export_absolute_times": "on"})
    exp.refresh_from_db()
    assert exp.use_default_settings is True
    assert exp.settings == {}


def test_default_experiment_settings_saves_global(client):
    client.post(reverse("web:default_experiment_settings"),
                {"export_absolute_times": ""})  # unchecked → False
    assert GlobalSettings.get_solo().default_experiment_settings == {"export_absolute_times": False}


def test_global_settings_page_links_to_defaults(client):
    body = client.get(reverse("web:global_settings")).content.decode()
    assert reverse("web:default_experiment_settings") in body


def test_experiment_page_has_settings_link(client):
    exp = _exp()
    body = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    assert reverse("web:experiment_settings", args=[exp.pk]) in body


def test_sidebar_has_global_settings_link(client):
    body = client.get(reverse("web:home")).content.decode()
    assert reverse("web:global_settings") in body
