"""The endpoint the metric switcher fetches from.

The detail page ships plot payloads for the metric it opens on only. Every
other metric comes from here, once each, the first time it is selected — the
same lazy-fetch-and-cache shape partial dependence and local ablation already
use (test_partial_dependence_view.py, test_trial_ablation.py).

Unlike those two this computes nothing new: the numbers were all worked out at
run completion and stored, so what is deferred is building the Plotly objects
and serializing them. That was 72 `plot()` calls and ~300KB of JSON on every
page load, for six payloads that end up on screen.
"""

import pytest
from django.urls import reverse

from core import io
from ui.models import GlobalSettings

from tests.conftest import FIXTURES_DIR


def _experiment_with_result():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _url(pk, metric):
    return reverse("ui:metric_figures", args=[pk]) + f"?metric={metric}"


def test_it_answers_with_every_per_metric_figure(client):
    exp = _experiment_with_result()
    payload = client.get(_url(exp.pk, "f1")).json()

    assert set(payload) == {
        "hyperparameter_importance",
        "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
        "interactions_coalitions", "interactions_orders",
        "performance_over_time", "configuration_cube", "parallel_coordinates",
        "partial_dependence", "acquisition_slice", "local_explanation",
        "local_effects",
    }


def test_it_omits_the_figures_that_are_not_per_metric(client):
    """Those are drawn once, from the page's own static payload — fetching them
    again per metric switch would be the waste this endpoint exists to end."""
    exp = _experiment_with_result()
    payload = client.get(_url(exp.pk, "f1")).json()

    assert "trial_duration" not in payload
    assert "trials" not in payload


def test_it_agrees_with_what_the_page_ships_for_its_own_metric(client):
    """The page and the switcher must not serve different shapes for the same
    figure — they go through one `_figure_plots`, and this is what pins that."""
    exp = _experiment_with_result()
    import json
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    start = html.index('id="metric-plots-data"')
    inline = json.loads(html[html.index(">", start) + 1: html.index("</script>", start)])

    fetched = client.get(_url(exp.pk, "accuracy")).json()

    assert fetched == inline["accuracy"]


def test_a_hidden_figure_is_absent_here_too(client):
    """The endpoint must not become a way to fetch what the settings switched
    off — it reads the same `_shown_figures` the page does."""
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {"show_configuration_cube": False}
    gs.save(update_fields=["default_experiment_settings"])

    exp = _experiment_with_result()
    payload = client.get(_url(exp.pk, "f1")).json()

    assert "configuration_cube" not in payload
    assert "parallel_coordinates" in payload, "the others are unaffected"


def test_rejects_unknown_metric(client):
    exp = _experiment_with_result()
    assert client.get(_url(exp.pk, "not-a-real-metric")).status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    assert client.get(_url(exp.pk, "accuracy")).status_code == 400


def test_unknown_experiment_404s(client):
    assert client.get(_url(999999, "accuracy")).status_code == 404
