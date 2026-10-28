"""`trial_panel` returns an HTML fragment with one trial's score, its delta to
the metric's best, and its configuration. In test2.ihpo, index 8 is best by
accuracy and index 0 scores 0.59375.
"""

import pytest
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment_with_result():
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _panel_url(pk, metric, idx):
    return reverse("ui:trial_panel", args=[pk]) + f"?metric={metric}&idx={idx}"


def test_selecting_the_best_trial_shows_a_zero_delta_marked_incumbent(client):
    """The best trial shows a zero delta marked `(Incumbent)`, which keeps the
    panel height stable.
    """
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 8))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Trial 9" in body
    assert "0.6813" in body or "0.68125" in body
    assert "metric-delta level" in body
    assert "(Incumbent)" in body


def test_selecting_a_non_best_trial_shows_a_negative_delta(client):
    """0.59375 minus the best 0.68125 gives a delta of about -0.0875.
    """
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 0))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Trial 1" in body
    assert "0.5938" in body or "0.59375" in body
    assert "metric-delta" in body
    assert "down" in body  # negative delta
    assert "0.0875" in body


def test_panel_includes_the_selected_trials_config(client):
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 8))
    body = resp.content.decode()

    # Random Forest hyperparameters
    for key in ("n_estimators", "max_depth", "min_samples_split", "max_features"):
        assert key in body


def test_panel_respects_the_requested_metric(client):
    exp = _experiment_with_result()
    resp_acc = client.get(_panel_url(exp.pk, "accuracy", 0))
    resp_f1 = client.get(_panel_url(exp.pk, "f1", 0))
    assert resp_acc.content != resp_f1.content


def test_rejects_unknown_metric(client):
    """An unknown metric gives 400, never 500."""
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "not-a-real-metric", 0))
    assert resp.status_code == 400


@pytest.mark.parametrize("idx", ["-1", "9999", "not-a-number"])
def test_rejects_out_of_range_or_malformed_index(idx):
    from django.test import Client
    exp = _experiment_with_result()
    resp = Client().get(_panel_url(exp.pk, "accuracy", idx))
    assert resp.status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    resp = client.get(_panel_url(exp.pk, "accuracy", 0))
    assert resp.status_code == 400


def test_unknown_experiment_404s(client):
    resp = client.get(_panel_url(999999, "accuracy", 0))
    assert resp.status_code == 404
