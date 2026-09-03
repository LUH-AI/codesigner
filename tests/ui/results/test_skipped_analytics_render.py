"""A run whose importance analytics were skipped still renders a usable page.

The eager games are skipped when a run is cancelled or when the model is too wide
to afford (see `tests/ui/runs/test_analytics_guard.py`). That leaves empty
importance dicts with a reason in the matching `_warning` field — the same shape
a HyperSHAP failure has always produced.

Two figures read importance for something other than drawing it, and both are
supposed to fall back rather than break: parallel coordinates orders its axes by
it, and each metric panel's default partial-dependence hyperparameter (`top_hp`)
is the most important one. These pin that, because "it degrades cleanly" was an
argument for not building any new UI to explain a skip, and an untested argument
is just a hope.
"""

import json

import pytest

from core import io
from ui.figures import parallel_coordinates_plot
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR

SKIP_REASON = ("Importance analytics were skipped: 10 hyperparameters over 4 "
               "metrics would need 12,288 coalition evaluations, past the 1,024 "
               "this deployment allows (ANALYTICS_EAGER_MAX_COALITIONS).")


def _experiment_with_skipped_analytics():
    """The Random Forest fixture, rewritten as though its analytics were skipped."""
    snapshot = io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes())
    metrics = snapshot["metrics"]["names"]
    result = snapshot["result"]
    for field in ("hyperparameter_importance", "hyperparameter_sensitivity",
                  "hyperparameter_mistunability", "hyperparameter_interactions"):
        result[field] = {m: {} for m in metrics}
        result[f"{field}_warning"] = {m: SKIP_REASON for m in metrics}
    return adapter.experiment_from_snapshot(snapshot)


def _rebuilt_result(exp):
    from ui.views import _rebuild_experiment
    return _rebuild_experiment(exp)["result"]


def test_the_detail_page_still_renders(client):
    exp = _experiment_with_skipped_analytics()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert resp.status_code == 200


def test_the_reason_reaches_the_page(client):
    """No new UI was built for this, on the argument that the existing per-metric
    warning channel already surfaces it. This is that argument, checked."""
    exp = _experiment_with_skipped_analytics()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert "coalition evaluations" in resp.content.decode()


def test_parallel_coordinates_falls_back_to_config_order(client):
    """With nothing to rank by, the axes keep the order the configurations use
    rather than collapsing or raising."""
    exp = _experiment_with_skipped_analytics()
    result = _rebuilt_result(exp)

    fig = parallel_coordinates_plot(result, "accuracy")

    assert fig is not None
    labels = list(fig.layout.xaxis.ticktext)
    hp_names = list(result.trials[0].config.keys())
    assert labels[:-1] == hp_names, "config order, unpermuted"
    assert len(labels) == len(hp_names) + 1, "plus the score axis"


def test_the_partial_dependence_default_falls_back_to_the_first_hyperparameter(client):
    """`top_hp` picks the most important hyperparameter; with no importance it
    has to pick *something* the picker can actually fetch."""
    exp = _experiment_with_skipped_analytics()
    resp = client.get(f"/experiments/{exp.pk}/")

    panels = resp.context["panels"]
    first_hp = list(_rebuilt_result(exp).trials[0].config.keys())[0]
    assert panels, "the page still builds its metric panels"
    for panel in panels:
        assert panel["top_hp"] == first_hp


def test_the_importance_figure_reports_no_data_rather_than_an_empty_chart(client):
    """Empty importance must not render as a chart of nothing."""
    exp = _experiment_with_skipped_analytics()
    resp = client.get(f"/experiments/{exp.pk}/")

    payload = resp.context["metric_plots"]["accuracy"]["hyperparameter_importance"]
    assert payload["tunability-pie"] is None
    assert payload["tunability-bar"] is None
