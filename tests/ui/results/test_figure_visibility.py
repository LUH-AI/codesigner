"""Which figures an experiment page shows is a setting.

A "figure" is one analytics panel on an experiment page. The seven of them are
declared once in `ui.figures.catalog`; each has a visibility setting, on by
default, editable on the default-experiment-settings page. Turning one off must
remove it from the page without disturbing the rest — the panels share a script,
so a hidden figure is exactly where a stale DOM lookup would break the others.
"""

from django.urls import reverse

from ui.figures import FIGURES, FIGURES_BY_KEY, FULL, HALF, Figure
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
    """An experiment with one finished trial, so every figure has data."""
    return Experiment.objects.create(
        name="figures", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], current_metric="accuracy",
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


def test_catalog_declares_the_seven_figures():
    """The catalog is the one list defining what a figure is; everything else
    (settings keys, the settings page, the detail page) reads it."""
    assert [c.key for c in FIGURES] == EXPECTED_KEYS


def test_every_figure_has_a_label_and_a_template():
    for figure in FIGURES:
        assert str(figure.label)
        assert figure.template == f"ui/figures/{figure.key}.html"
        assert figure.setting_key == f"show_{figure.key}"


def test_declaring_a_subclass_derives_everything_from_its_key():
    """Adding a figure is subclassing it: the template it renders, the setting
    that hides it and the id its plot mounts in all follow from the key, so a
    new figure needs no wiring anywhere else."""
    class Whatever(Figure):
        key = "whatever"
        label = "Whatever"

    assert Whatever.template == "ui/figures/whatever.html"
    assert Whatever.setting_key == "show_whatever"
    assert Whatever.dom_id == "figure-whatever"
    # unspecified display characteristics fall back to the common case
    assert Whatever.width == HALF
    assert Whatever.per_metric is False
    assert Whatever.absolute_scale is None
    assert Whatever.plot(result=None) is None


def test_each_figure_declares_its_width():
    """Width is declared per figure, and is one of the two supported layouts —
    a half-page tile or a full-page span."""
    assert {f.width for f in FIGURES} <= {HALF, FULL}
    assert FIGURES_BY_KEY["trials"].width == FULL
    assert FIGURES_BY_KEY["trial_duration"].width == HALF


def test_only_metric_dependent_figures_are_marked_per_metric():
    """Per-metric figures are rebuilt when the metric changes; trial duration
    is the same plot for every metric, and the tables aren't plots at all."""
    per_metric = {f.key for f in FIGURES if f.per_metric}
    assert per_metric == {"hyperparameter_importance", "incumbent_performance",
                          "error_over_time"}


def test_the_scale_toggle_is_declared_not_hardcoded():
    """A figure opts into the absolute/relative y-scale button by declaring the
    range its 'absolute' means; the page builds the button from that."""
    assert FIGURES_BY_KEY["incumbent_performance"].absolute_scale == {
        "yaxis.range": [0, 1], "yaxis.autorange": False}
    # the error figure's y-axis is log, so its range is in log10 units
    assert FIGURES_BY_KEY["error_over_time"].absolute_scale == {
        "yaxis.range": [-3, 0], "yaxis.autorange": False}
    assert FIGURES_BY_KEY["trials"].absolute_scale is None


def test_all_figures_show_by_default(client):
    html = _page(client, _experiment())
    for figure in FIGURES:
        assert f'data-figure="{figure.key}"' in html, figure.key


def test_unchecking_a_figure_removes_it_from_the_page(client):
    _hide("hyperparameter_importance")
    html = _page(client, _experiment())

    assert 'data-figure="hyperparameter_importance"' not in html
    # the others are untouched
    assert 'data-figure="incumbent_performance"' in html
    assert 'data-figure="trials"' in html


def test_hiding_the_performance_figure_keeps_the_page_working(client):
    """The performance figure owns click-to-select, so the script reaches for it
    by id. Hidden, the page must still render its remaining figures."""
    _hide("incumbent_performance")
    html = _page(client, _experiment())

    assert 'data-figure="incumbent_performance"' not in html
    assert 'id="figure-incumbent_performance"' not in html
    assert 'data-figure="error_over_time"' in html
    assert 'data-figure="best_configuration"' in html


def test_all_figures_can_be_hidden_at_once(client):
    _hide(*EXPECTED_KEYS)
    html = _page(client, _experiment())

    for figure in FIGURES:
        assert f'data-figure="{figure.key}"' not in html, figure.key
    assert "Trials" not in html.split("<main", 1)[-1]


def test_a_hidden_figure_ships_no_plot_data(client):
    """A figure that is off is not computed either — the JSON exists only to be
    drawn, so an absent figure should leave nothing behind in it."""
    _hide("trial_duration", "error_over_time")
    html = _page(client, _experiment())
    static = html.split('id="static-plots-data"', 1)[1].split("</script>", 1)[0]
    per_metric = html.split('id="metric-plots-data"', 1)[1].split("</script>", 1)[0]

    assert "trial_duration" not in static
    assert "error_over_time" not in per_metric
    # the figures still on the page are unaffected
    assert "incumbent_performance" in per_metric
