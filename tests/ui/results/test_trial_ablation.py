"""The local explanation depends on the selected trial, so `trial_ablation`
computes it on request.
"""

import pytest
from django.urls import reverse

from core import io
from ui.views.analytics import _local_ablation_data

from tests.conftest import FIXTURES_DIR


def test_no_model_reports_unavailable_instead_of_crashing():
    """Without a model the view returns a warning instead of a 500."""
    figure, warning, _effects = _local_ablation_data({"model": None}, "accuracy", 0)
    assert figure is None
    assert warning and "model" in warning.lower()


def _experiment_with_result():
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _ablation_url(pk, metric, idx):
    return reverse("ui:trial_ablation", args=[pk]) + f"?metric={metric}&idx={idx}"


def test_explains_a_trial_against_the_default(client):
    """The registry model resolves even read-only, so no warning appears."""
    exp = _experiment_with_result()
    resp = client.get(_ablation_url(exp.pk, "accuracy", 8))

    assert resp.status_code == 200
    data = resp.json()
    assert data["warning"] is None
    assert data["figure"]["data"][0]["type"] == "waterfall"


def test_different_trials_explain_differently(client):
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
    from experiments.services import snapshot as adapter

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
