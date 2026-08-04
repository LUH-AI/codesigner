"""The two settings pages are the same page, differing only in scope.

The defaults page edits the template experiments follow; an experiment's own
page edits its copy of exactly the same settings, so both render one shared
partial. What only the per-experiment page has is its relationship to the
defaults: the inherit checkbox, the three buttons, and a way back to the
experiment it belongs to.
"""

from django.urls import reverse

from ui.figures import FIGURES
from ui.models import Experiment, GlobalSettings
from ui.services.settings import SETTING_DEFAULTS


def _exp():
    return Experiment.objects.create(
        name="s", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def _both_pages(client):
    exp = _exp()
    return (client.get(reverse("ui:default_experiment_settings")).content.decode(),
            client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode(),
            exp)


def test_both_pages_offer_the_same_settings(client):
    """Every setting in the schema has a control on both pages."""
    defaults, per_experiment, _ = _both_pages(client)
    for key in SETTING_DEFAULTS:
        assert f'name="{key}"' in defaults, key
        assert f'name="{key}"' in per_experiment, key


def test_both_pages_label_the_figures_identically(client):
    defaults, per_experiment, _ = _both_pages(client)
    for figure in FIGURES:
        assert str(figure.label) in defaults, figure.key
        assert str(figure.label) in per_experiment, figure.key


def test_only_the_experiment_page_has_the_inherit_toggle(client):
    defaults, per_experiment, _ = _both_pages(client)
    assert 'name="use_default_settings"' in per_experiment
    assert 'name="use_default_settings"' not in defaults


def test_experiment_page_has_the_three_buttons(client):
    _, per_experiment, _ = _both_pages(client)
    assert 'name="reset"' in per_experiment
    assert 'name="save_as_default"' in per_experiment
    assert "Save settings" in per_experiment


def test_experiment_page_links_back_to_the_experiment(client):
    _, per_experiment, exp = _both_pages(client)
    assert reverse("ui:experiment_detail", args=[exp.pk]) in per_experiment


def test_inherited_settings_are_shown_but_marked_unusable(client):
    """Inheriting shows the settings (so you can see them) in a pane the page
    greys out and disables, rather than hiding them."""
    _, per_experiment, _ = _both_pages(client)
    assert 'data-pane="overrides"' in per_experiment
    assert "pane-disabled" in per_experiment
    assert "el.disabled = cb.checked" in per_experiment


def test_saving_an_override_records_every_setting(client):
    """An experiment's own settings are a full copy, not a partial patch."""
    exp = _exp()
    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"export_absolute_times": "on", "show_trials": "on"})
    exp.refresh_from_db()

    assert exp.use_default_settings is False
    assert set(exp.settings) == set(SETTING_DEFAULTS)
    assert exp.settings["export_absolute_times"] is True
    assert exp.settings["show_trials"] is True
    assert exp.settings["show_trial_duration"] is False   # box wasn't ticked


def test_reset_restores_inheritance(client):
    exp = _exp()
    exp.use_default_settings = False
    exp.settings = {"export_absolute_times": False}
    exp.save()
    client.post(reverse("ui:experiment_settings", args=[exp.pk]), {"reset": "1"})
    exp.refresh_from_db()

    assert exp.use_default_settings is True
    assert exp.settings == {}


def test_save_as_default_asks_before_changing_anything(client):
    """Promoting one experiment's settings changes what every inheriting
    experiment shows, so it confirms first and writes nothing yet."""
    exp = _exp()
    resp = client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                       {"save_as_default": "1", "show_trials": "on"})

    assert resp.status_code == 200
    assert 'name="confirm_save_as_default"' in resp.content.decode()
    assert GlobalSettings.get_solo().default_experiment_settings == {}


def test_confirming_save_as_default_writes_the_defaults(client):
    exp = _exp()
    resp = client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                       {"save_as_default": "1", "show_trials": "on"})
    # the confirmation carries the submitted settings forward
    assert 'name="show_trials"' in resp.content.decode()

    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"confirm_save_as_default": "1", "show_trials": "on"})

    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["show_trials"] is True
    assert stored["show_trial_duration"] is False
    assert set(stored) == set(SETTING_DEFAULTS)


def test_save_as_default_leaves_the_experiment_alone(client):
    """It changes the defaults, not this experiment's own relationship to them."""
    exp = _exp()
    client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                {"confirm_save_as_default": "1", "show_trials": "on"})
    exp.refresh_from_db()

    assert exp.use_default_settings is True
    assert exp.settings == {}
