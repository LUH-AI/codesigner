"""The default-experiment-settings page controls which figures appear.

One checkbox per figure in `ui.figures.catalog`, checked by default. Saving writes
a visibility flag per figure into the global default experiment settings, which is
what an experiment page reads when deciding what to render.
"""

from django.urls import reverse

from ui.figures import FIGURES
from ui.models import GlobalSettings
from ui.services.settings import SETTING_DEFAULTS, global_defaults


def test_every_figure_has_a_visibility_setting_defaulting_to_on():
    for figure in FIGURES:
        assert SETTING_DEFAULTS[figure.setting_key] is True
        assert global_defaults()[figure.setting_key] is True


def test_page_renders_a_checkbox_per_figure(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for figure in FIGURES:
        assert f'name="{figure.setting_key}"' in body, figure.key


def test_checkboxes_start_checked_when_no_choice_is_stored(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    # each figure's checkbox input carries `checked`
    for figure in FIGURES:
        field = body.split(f'name="{figure.setting_key}"', 1)[1].split(">", 1)[0]
        assert "checked" in field, figure.key


def test_saving_records_each_figure_choice(client):
    """Posting with one figure's box unticked stores that one off and the rest on."""
    kept = [c for c in FIGURES if c.key != "trial_duration"]
    client.post(reverse("ui:default_experiment_settings"),
                {
                 **{c.setting_key: "on" for c in kept}})

    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["show_trial_duration"] is False
    for figure in kept:
        assert stored[figure.setting_key] is True


def test_a_stored_choice_comes_back_unchecked(client):
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_trials": False}
    gs.save(update_fields=["default_experiment_settings"])

    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    field = body.split('name="show_trials"', 1)[1].split(">", 1)[0]
    assert "checked" not in field
