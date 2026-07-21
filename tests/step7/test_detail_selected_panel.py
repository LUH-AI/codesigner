"""Step 7: the detail page embeds a default selected-config panel per metric,
and wires up click-to-select on the performance chart.

Mirrors app/experiment.py's default (sel_key defaults to the metric's best
trial until a point is clicked) and the on_select handler in
app/analytics/performance.py. The actual click/fetch/restyle behavior runs in
the browser and isn't exercised by these server-rendered-HTML tests; it's
covered by the trial_panel view tests plus a manual browser check.
"""

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _detail_html(client):
    from web.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    return client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode(), exp


def test_detail_has_a_selected_config_panel_per_metric(client):
    """Each metric gets a 'Selected configuration' panel, like best-config."""
    html, exp = _detail_html(client)
    assert "Selected configuration" in html
    for m in exp.metric_names:
        assert f'class="card panel selected-config" data-metric="{m}"' in html \
            or f'data-metric="{m}"' in html


def test_default_selection_matches_the_best_trial_with_no_delta(client):
    """Before any click, the selected-config panel shows the best trial (trial 9
    for accuracy in the fixture) and no delta, matching the best-config panel.

    Distinguishes the selected-config panel from best-config by its distinct
    caption ("click any point on the graph to select"), so this can't pass
    just because the best-config panel happens to mention trial 9.
    """
    html, _ = _detail_html(client)
    assert "click any point on the graph to select" in html
    idx = html.index("click any point on the graph to select")
    window = html[max(0, idx - 400):idx + 400]
    assert "Trial 9" in window
    assert "0.6813" in window or "0.68125" in window
    assert "metric-delta" not in window


def test_detail_wires_up_click_to_select(client):
    """The page attaches a click handler to the performance chart and points it
    at the trial-panel endpoint."""
    html, exp = _detail_html(client)
    assert "plotly_click" in html
    assert reverse("web:trial_panel", args=[exp.pk]) in html
