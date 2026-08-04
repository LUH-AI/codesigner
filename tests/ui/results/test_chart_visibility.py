"""Which charts an experiment page shows is a setting.

A "chart" is one analytics panel on an experiment page. The seven of them are
declared once in `ui.charts.catalog`; each has a visibility setting, on by
default, editable on the default-experiment-settings page. Turning one off must
remove it from the page without disturbing the rest — the panels share a script,
so a hidden chart is exactly where a stale DOM lookup would break the others.
"""

from django.urls import reverse

from ui.charts import CHARTS
from ui.models import Experiment, GlobalSettings

EXPECTED_KEYS = [
    "best_configuration",
    "selected_configuration",
    "hyperparameter_importance",
    "incumbent_performance",
    "trial_duration",
    "error_over_time",
    "trials",
]


def _experiment():
    """An experiment with one finished trial, so every chart has data."""
    return Experiment.objects.create(
        name="charts", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], primary_metric="accuracy",
        original_metric="accuracy", seed=0,
        result={
            "stats": {"submitted": 1, "finished": 1, "running": 0},
            "data": [{"config_id": 1, "cost": 0.2, "time": 2.5,
                      "scores": {"accuracy": 0.8, "f1": 0.7},
                      "incumbent_score": 0.8, "incumbent_config_id": 1}],
            "configs": {"1": {"n_estimators": 100}},
            "config_origins": {"1": "Random Search"}, "optimizer_state": {},
            "primary_metric": "accuracy", "best_score": 0.8, "best_config_id": "1",
            "hyperparameter_importance": {"accuracy": {"n_estimators": 1.0}},
            "hyperparameter_importance_warning": {}, "trials_limit": None,
        },
    )


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def _hide(*keys):
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {f"show_{k}": False for k in keys}
    gs.save(update_fields=["default_experiment_settings"])


def test_catalog_declares_the_seven_charts():
    """The catalog is the one list defining what a chart is; everything else
    (settings keys, the settings page, the detail page) reads it."""
    assert [c.key for c in CHARTS] == EXPECTED_KEYS


def test_every_chart_has_a_label_and_a_template():
    for chart in CHARTS:
        assert str(chart.label)
        assert chart.template == f"ui/charts/{chart.key}.html"
        assert chart.setting_key == f"show_{chart.key}"


def test_all_charts_show_by_default(client):
    html = _page(client, _experiment())
    for chart in CHARTS:
        assert f'data-chart="{chart.key}"' in html, chart.key


def test_unchecking_a_chart_removes_it_from_the_page(client):
    _hide("hyperparameter_importance")
    html = _page(client, _experiment())

    assert 'data-chart="hyperparameter_importance"' not in html
    # the others are untouched
    assert 'data-chart="incumbent_performance"' in html
    assert 'data-chart="trials"' in html


def test_hiding_the_performance_chart_keeps_the_page_working(client):
    """The performance chart owns click-to-select, so the script reaches for it
    by id. Hidden, the page must still render its remaining charts."""
    _hide("incumbent_performance")
    html = _page(client, _experiment())

    assert 'data-chart="incumbent_performance"' not in html
    assert 'id="chart-incumbent_performance"' not in html
    assert 'data-chart="error_over_time"' in html
    assert 'data-chart="best_configuration"' in html


def test_all_charts_can_be_hidden_at_once(client):
    _hide(*EXPECTED_KEYS)
    html = _page(client, _experiment())

    for chart in CHARTS:
        assert f'data-chart="{chart.key}"' not in html, chart.key
    assert "Trials" not in html.split("<main", 1)[-1]


def test_a_hidden_chart_is_not_serialized_into_the_page(client):
    """A chart that is off should not ship its figure JSON either — the data is
    only there to be drawn."""
    _hide("trial_duration")
    html = _page(client, _experiment())

    assert 'id="trial-duration-data"' not in html
