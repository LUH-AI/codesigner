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
    assert 'id="figure-incumbent_performance"' in html
    assert 'id="figure-hyperparameter_importance"' in html


def test_detail_has_metric_switcher_with_all_metrics(client):
    """A metric selector offers every scored metric."""
    html, exp = _detail(client)
    assert 'id="metric-select"' in html
    for m in exp.metric_names:
        assert f'value="{m}"' in html


def test_embedded_figures_cover_every_metric(client):
    """Every metric's performance figure is embedded for the client-side switcher."""
    html, exp = _detail(client)
    start = html.index('id="metric-plots-data"')
    payload = html[html.index(">", start) + 1: html.index("</script>", start)]
    plots = json.loads(payload)
    for m in exp.metric_names:
        assert m in plots
        assert plots[m]["incumbent_performance"]["data"]


def test_best_config_panel_present_per_metric(client):
    """Each metric gets its own best-configuration panel to toggle between."""
    html, exp = _detail(client)
    for m in exp.metric_names:
        assert f'data-metric="{m}"' in html
