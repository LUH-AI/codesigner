"""The `partial_dependence` view builds each figure on request, because the
hyperparameter to explain is a picker choice.
"""

import pytest
from django.urls import reverse

from core import io
from ui.views.analytics import _partial_dependence_data

from tests.conftest import FIXTURES_DIR


def test_no_model_reports_unavailable_instead_of_crashing():
    """Without a model the view returns a warning instead of a 500."""
    figure, warning = _partial_dependence_data({"model": None}, "accuracy", "max_depth")
    assert figure is None
    assert warning and "model" in warning.lower()


def _experiment_with_result():
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _pdp_url(pk, metric, hp):
    return reverse("ui:partial_dependence", args=[pk]) + f"?metric={metric}&hp={hp}"


def test_explains_one_hyperparameters_partial_dependence(client):
    exp = _experiment_with_result()
    resp = client.get(_pdp_url(exp.pk, "accuracy", "max_depth"))

    assert resp.status_code == 200
    data = resp.json()
    assert data["warning"] is None
    assert len(data["figure"]["data"]) == 2
    # The trace names are translatable, so the test tells ICE from PDP by
    # opacity and mode.
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
    from experiments.services import snapshot as adapter

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
