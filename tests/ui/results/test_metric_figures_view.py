"""The metric switcher fetches each other metric here once. The numbers are
stored at run completion, so the endpoint only defers building and serializing
the Plotly objects.
"""

import pytest
from django.urls import reverse

from core import io
from experiments.models import GlobalSettings

from tests.conftest import FIXTURES_DIR


def _experiment_with_result():
    from experiments.services import snapshot as adapter
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
    exp = _experiment_with_result()
    payload = client.get(_url(exp.pk, "f1")).json()

    assert "trial_duration" not in payload
    assert "trials" not in payload


def test_it_agrees_with_what_the_page_ships_for_its_own_metric(client):
    """Both paths go through one `_figure_plots`, so page and switcher serve the
    same shape."""
    exp = _experiment_with_result()
    import json
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    start = html.index('id="metric-plots-data"')
    inline = json.loads(html[html.index(">", start) + 1: html.index("</script>", start)])

    fetched = client.get(_url(exp.pk, "accuracy")).json()

    assert fetched == inline["accuracy"]


def test_a_hidden_figure_is_absent_here_too(client):
    """The endpoint reads the same `_shown_figures` as the page."""
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
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    assert client.get(_url(exp.pk, "accuracy")).status_code == 400


def test_unknown_experiment_404s(client):
    assert client.get(_url(999999, "accuracy")).status_code == 404
