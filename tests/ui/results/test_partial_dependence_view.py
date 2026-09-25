"""The partial-dependence endpoint backing the "Partial dependence (PDP/ICE)"
figure.

Unlike the three global HyperSHAP games (computed once at run completion and
stored on the result), which hyperparameter is being explained is a picker
choice, not one of a small precomputable set — so it is fetched here,
lazily, the same way trial_ablation already fetches the importance figure's
"Local" game. Uses the same test2.ihpo fixture (Random Forest, 30 trials):
max_depth/max_features/min_samples_split/n_estimators are its hyperparameters.
"""

import pytest
from django.urls import reverse

from core import io
from ui.views import _partial_dependence_data

from tests.conftest import FIXTURES_DIR


def test_no_model_reports_unavailable_instead_of_crashing():
    """A custom-model experiment viewed read-only has no model to build a
    config space (or fit a surrogate) from — a plain warning, not a 500."""
    figure, warning = _partial_dependence_data({"model": None}, "accuracy", "max_depth")
    assert figure is None
    assert warning and "model" in warning.lower()


def _experiment_with_result():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _pdp_url(pk, metric, hp):
    return reverse("ui:partial_dependence", args=[pk]) + f"?metric={metric}&hp={hp}"


def test_explains_one_hyperparameters_partial_dependence(client):
    """A valid (metric, hyperparameter) request returns a real Plotly figure
    with both an ICE trace and a PDP trace — the model resolves for this
    registry-model fixture even read-only, so there's no "model unavailable"
    warning here."""
    exp = _experiment_with_result()
    resp = client.get(_pdp_url(exp.pk, "accuracy", "max_depth"))

    assert resp.status_code == 200
    data = resp.json()
    assert data["warning"] is None
    assert len(data["figure"]["data"]) == 2
    # Told apart by what they are, not by what they are called — those names are
    # translatable and a rewording should not fail a test about structure. The
    # per-trial curves are the faint ones; the average is the marked line.
    ice, pdp = data["figure"]["data"]
    assert ice["name"] and pdp["name"] and ice["name"] != pdp["name"]
    assert ice["opacity"] < 0.5, "the individual trials are the faint ones"
    assert "markers" in pdp["mode"], "the average is the marked line"


def test_different_hyperparameters_explain_differently(client):
    exp = _experiment_with_result()
    a = client.get(_pdp_url(exp.pk, "accuracy", "max_depth")).json()
    b = client.get(_pdp_url(exp.pk, "accuracy", "n_estimators")).json()
    assert a["figure"]["data"][1]["y"] != b["figure"]["data"][1]["y"]


def test_rejects_unknown_metric(client):
    exp = _experiment_with_result()
    resp = client.get(_pdp_url(exp.pk, "not-a-real-metric", "max_depth"))
    assert resp.status_code == 400


def test_rejects_unknown_hyperparameter(client):
    exp = _experiment_with_result()
    resp = client.get(_pdp_url(exp.pk, "accuracy", "not-a-real-hyperparameter"))
    assert resp.status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    resp = client.get(_pdp_url(exp.pk, "accuracy", "max_depth"))
    assert resp.status_code == 400


def test_unknown_experiment_404s(client):
    resp = client.get(_pdp_url(999999, "accuracy", "max_depth"))
    assert resp.status_code == 404
