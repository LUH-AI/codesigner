"""Step 4: the results page renders charts and a metric switcher.

Drives the run view and checks the results page ships everything the browser
needs to draw and switch charts: the vendored Plotly script, a metric
selector, chart containers, and embedded per-metric figure JSON.
"""

import json

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR


def _run(client, **overrides):
    data = {
        "name": "charts-run",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "primary_metric": "accuracy",
        "seed": 0,
        "n_trials": 3,
    }
    data.update(overrides)
    return client.post(reverse("web:new_experiment"), data)


def test_results_page_includes_plotly_and_chart_containers(client):
    """The results page loads Plotly and provides the chart mount points."""
    html = _run(client).content.decode()
    assert "plotly.min.js" in html
    assert 'id="perf-chart"' in html
    assert 'id="imp-chart"' in html


def test_results_page_has_metric_switcher_with_all_metrics(client):
    """A metric selector offers every scored metric, defaulting to the optimized one.

    Expect: a <select> with an <option> for each of the four metrics, and the
    optimized metric (accuracy) preselected.
    """
    html = _run(client).content.decode()
    assert 'id="metric-select"' in html
    for m in ("accuracy", "f1", "precision", "recall(macro)"):
        assert f'value="{m}"' in html
    assert 'value="accuracy" selected' in html


def test_embedded_figures_cover_every_metric(client):
    """The embedded figure JSON contains a performance figure for each metric.

    The switcher works client-side, so all metrics' figures must be present in
    the page. Parse the json_script payload and check each metric has a
    performance figure with data.
    """
    html = _run(client).content.decode()
    start = html.index('id="figures-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    figures = json.loads(payload)

    for m in ("accuracy", "f1", "precision", "recall(macro)"):
        assert m in figures
        assert figures[m]["performance"]["data"]


def test_best_config_panel_present_per_metric(client):
    """Each metric gets its own best-configuration panel to toggle between."""
    html = _run(client).content.decode()
    for m in ("accuracy", "f1", "precision", "recall(macro)"):
        assert f'data-metric="{m}"' in html
