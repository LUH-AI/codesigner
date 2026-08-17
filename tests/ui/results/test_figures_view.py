"""Result figures render on the detail page.

Figures used to be checked on the synchronous new_experiment response; with
create and run separated, they render on an experiment's detail page once it
has a result. We set up an experiment carrying a stored result (via the
adapter) and GET its detail page.

DB access + isolated media come from tests/ui/conftest.py.
"""

import json

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _detail(client):
    """Create an experiment with the 30-trial fixture result; return (html, exp)."""
    from ui.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html, exp


def test_detail_includes_plotly_and_chart_containers(client):
    """The detail page loads Plotly and provides the figure mount points."""
    html, _ = _detail(client)
    assert "plotly.min.js" in html
    assert 'id="figure-performance_over_time"' in html
    assert 'id="figure-hyperparameter_importance"' in html


def test_detail_has_metric_switcher_with_all_metrics(client):
    """A metric selector offers every scored metric."""
    html, exp = _detail(client)
    assert 'id="metric-select"' in html
    for m in exp.metric_names:
        assert f'value="{m}"' in html


def test_embedded_figures_cover_every_metric(client):
    """Every metric's performance figure is embedded for the client-side
    switcher — one entry per declared view, the default (trial-score) among
    them."""
    html, exp = _detail(client)
    start = html.index('id="metric-plots-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    plots = json.loads(payload)
    for m in exp.metric_names:
        assert m in plots
        assert plots[m]["performance_over_time"]["trial-score"]["data"]


def test_best_config_panel_present_per_metric(client):
    """Each metric gets its own best-configuration panel to toggle between."""
    html, exp = _detail(client)
    for m in exp.metric_names:
        assert f'data-metric="{m}"' in html


def test_performance_over_time_offers_all_four_views(client):
    """Trial/time x score/error compose to four views, each carrying data."""
    html, exp = _detail(client)
    assert 'id="view-select-performance_over_time-x"' in html
    assert 'id="view-select-performance_over_time-y"' in html

    start = html.index('id="metric-plots-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    plots = json.loads(payload)
    for m in exp.metric_names:
        views = plots[m]["performance_over_time"]
        assert set(views) == {"trial-score", "trial-error", "time-score", "time-error"}
        for view in views.values():
            assert view["data"]


def test_hyperparameter_importance_offers_three_games_and_three_renderings(client):
    """Tunability, sensitivity and mistunability x pie/bar/table — 9 flat view
    keys. pie/bar are plot payloads, table is None (the template renders it
    directly) but the table markup itself is present, one block per metric
    per game."""
    html, exp = _detail(client)
    assert 'id="view-select-hyperparameter_importance-game"' in html
    assert 'id="view-select-hyperparameter_importance-rendering"' in html

    start = html.index('id="metric-plots-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    plots = json.loads(payload)
    games = ("tunability", "sensitivity", "mistunability")
    for m in exp.metric_names:
        views = plots[m]["hyperparameter_importance"]
        assert set(views) == {f"{g}-{r}" for g in games for r in ("pie", "bar", "table")}
        for game in games:
            assert views[f"{game}-table"] is None
            assert f'class="importance-table" data-metric="{m}" data-game="{game}"' in html
        # tunability always has real data on this fixture; sensitivity and
        # mistunability's pie/bar payloads at least round-trip (possibly
        # empty on a fixture that predates them).
        assert views["tunability-pie"]["data"]
        assert views["tunability-bar"]["data"]


def test_table_view_suppresses_the_generic_empty_message(client):
    """The table view's payload is None by design (the template renders it
    directly, not from a plot) — the same signal draw() otherwise reads as
    "no data" and shows the empty-message caption for. A browser check caught
    this actually happening (the caption showing over a populated table); the
    JS-side fix isn't executed by this suite, so this pins the wiring the way
    test_scale_button.py already does for other client-side behavior."""
    html, _ = _detail(client)
    assert "syncImportanceTable" in html
    assert "figure-hyperparameter_importance-empty" in html
    script = html[html.index("function syncImportanceTable"):]
    script = script[:script.index("function show(metric)")]
    assert "empty.hidden = true" in script


def test_figure_views_lists_only_multiview_figures(client):
    """Single-view figures (trial_duration, the tables) never appear here —
    the script uses this to tell a plain payload from a keyed-by-view one
    without guessing from its shape."""
    html, _ = _detail(client)
    start = html.index('id="figure-views-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    views = json.loads(payload)

    assert set(views) == {
        "hyperparameter_importance", "hyperparameter_interactions", "performance_over_time",
    }
    assert views["hyperparameter_importance"] == [
        "tunability-pie", "tunability-bar", "tunability-table",
        "sensitivity-pie", "sensitivity-bar", "sensitivity-table",
        "mistunability-pie", "mistunability-bar", "mistunability-table",
        "local-bar",
    ]
    assert views["hyperparameter_interactions"] == ["heatmap", "bar"]
    assert views["performance_over_time"] == [
        "trial-score", "trial-error", "time-score", "time-error"]


def test_hyperparameter_interactions_offers_heatmap_and_bar(client):
    """Both views come straight from the stored result — no lazy fetch like
    the importance figure's "local" game. The fixture predates this field
    (hyperparameter_interactions defaults to {} on deserialize), so both
    views round-trip as None here; test_plots.py covers the actual rendering
    against a populated dict."""
    html, exp = _detail(client)
    assert 'id="view-select-hyperparameter_interactions"' in html

    start = html.index('id="metric-plots-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    plots = json.loads(payload)
    for m in exp.metric_names:
        assert set(plots[m]["hyperparameter_interactions"]) == {"heatmap", "bar"}
