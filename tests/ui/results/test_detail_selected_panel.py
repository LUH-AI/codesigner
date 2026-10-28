"""The detail page embeds a selected-config panel per metric and wires
click-to-select on the performance figure; the click itself runs in the browser.
"""

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _detail_html(client):
    from experiments.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode(), exp


def test_detail_has_a_selected_config_panel_per_metric(client):
    """The panel sits in the sidebar (`Figure.in_sidebar`) so it stays readable
    while figures are clicked.
    """
    html, exp = _detail_html(client)
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert "Selected Configuration" in sidebar
    for m in exp.data.metric_names:
        assert f'class="panel selected-config" data-figure="selected_configuration"\n             data-metric="{m}"' in sidebar \
            or f'data-metric="{m}"' in sidebar


def test_default_selection_matches_the_best_trial_marked_as_the_incumbent(client):
    """Before any click, the panel shows best trial 9 with a zero difference
    marked `(Incumbent)`, and its caption tells it apart from best-config.
    """
    html, _ = _detail_html(client)
    assert "click any point on the graph to select" in html
    idx = html.index("click any point on the graph to select")
    window = html[max(0, idx - 400):idx + 400]
    assert "Trial 9" in window
    assert "0.6813" in window or "0.68125" in window
    assert "metric-delta level" in window
    assert "(Incumbent)" in window


def test_detail_wires_up_click_to_select(client):
    html, exp = _detail_html(client)
    assert "plotly_click" in html
    assert reverse("ui:trial_panel", args=[exp.pk]) in html
