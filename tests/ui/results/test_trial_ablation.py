"""The trial-ablation endpoint backing the importance figure's "Local
(selected trial)" game.

Unlike the other three HyperSHAP games (tunability/sensitivity/mistunability,
computed once at run completion and stored on the result), a local
explanation depends on which trial is selected — so it is fetched here,
lazily, the same way click-to-select already fetches the selected-config
panel (test_trial_panel.py). Uses the same test2.ihpo fixture (Random Forest,
30 trials): trial 9 (index 8) is the best trial by accuracy.
"""

import pytest
from django.urls import reverse

from core import io
from ui.views import _local_ablation_data

from tests.conftest import FIXTURES_DIR


def test_no_model_reports_unavailable_instead_of_crashing():
    """A custom-model experiment viewed read-only has no model to build a
    config space from — a plain warning, not a 500, and no attempt at a
    RandomForest-fallback style guess (there is nothing to guess from)."""
    figure, warning, _effects = _local_ablation_data({"model": None}, "accuracy", 0)
    assert figure is None
    assert warning and "model" in warning.lower()


def _experiment_with_result():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _ablation_url(pk, metric, idx):
    return reverse("ui:trial_ablation", args=[pk]) + f"?metric={metric}&idx={idx}"


def test_explains_a_trial_against_the_default(client):
    """A valid (metric, trial) request returns a real Plotly figure — the
    model resolves for this registry-model fixture even read-only, so there's
    no "model unavailable" warning here."""
    exp = _experiment_with_result()
    resp = client.get(_ablation_url(exp.pk, "accuracy", 8))

    assert resp.status_code == 200
    data = resp.json()
    assert data["warning"] is None
    assert data["figure"]["data"][0]["type"] == "waterfall"


def test_different_trials_explain_differently(client):
    """Two different trials' ablations aren't just copies of each other."""
    exp = _experiment_with_result()
    best = client.get(_ablation_url(exp.pk, "accuracy", 8)).json()
    other = client.get(_ablation_url(exp.pk, "accuracy", 0)).json()
    assert best["figure"]["data"][0]["y"] != other["figure"]["data"][0]["y"]


def test_rejects_unknown_metric(client):
    exp = _experiment_with_result()
    resp = client.get(_ablation_url(exp.pk, "not-a-real-metric", 0))
    assert resp.status_code == 400


@pytest.mark.parametrize("idx", ["-1", "9999", "not-a-number"])
def test_rejects_out_of_range_or_malformed_index(idx):
    from django.test import Client
    exp = _experiment_with_result()
    resp = Client().get(_ablation_url(exp.pk, "accuracy", idx))
    assert resp.status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    resp = client.get(_ablation_url(exp.pk, "accuracy", 0))
    assert resp.status_code == 400


def test_unknown_experiment_404s(client):
    resp = client.get(_ablation_url(999999, "accuracy", 0))
    assert resp.status_code == 404
