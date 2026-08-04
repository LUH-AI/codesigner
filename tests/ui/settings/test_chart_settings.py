"""The default-experiment-settings page controls which charts appear.

One checkbox per chart in `ui.charts.catalog`, checked by default. Saving writes
a visibility flag per chart into the global default experiment settings, which is
what an experiment page reads when deciding what to render.
"""

from django.urls import reverse

from ui.charts import CHARTS
from ui.models import GlobalSettings
from ui.services.settings import SETTING_DEFAULTS, global_defaults


def test_every_chart_has_a_visibility_setting_defaulting_to_on():
    for chart in CHARTS:
        assert SETTING_DEFAULTS[chart.setting_key] is True
        assert global_defaults()[chart.setting_key] is True


def test_page_renders_a_checkbox_per_chart(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    for chart in CHARTS:
        assert f'name="{chart.setting_key}"' in body, chart.key


def test_checkboxes_start_checked_when_no_choice_is_stored(client):
    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    # each chart's checkbox input carries `checked`
    for chart in CHARTS:
        field = body.split(f'name="{chart.setting_key}"', 1)[1].split(">", 1)[0]
        assert "checked" in field, chart.key


def test_saving_records_each_chart_choice(client):
    """Posting with one chart's box unticked stores that one off and the rest on."""
    kept = [c for c in CHARTS if c.key != "trial_duration"]
    client.post(reverse("ui:default_experiment_settings"),
                {"export_absolute_times": "on",
                 **{c.setting_key: "on" for c in kept}})

    stored = GlobalSettings.get_solo().default_experiment_settings
    assert stored["show_trial_duration"] is False
    for chart in kept:
        assert stored[chart.setting_key] is True


def test_a_stored_choice_comes_back_unchecked(client):
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_trials": False}
    gs.save(update_fields=["default_experiment_settings"])

    body = client.get(reverse("ui:default_experiment_settings")).content.decode()
    field = body.split('name="show_trials"', 1)[1].split(">", 1)[0]
    assert "checked" not in field
