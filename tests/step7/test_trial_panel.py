"""Step 7: the trial-panel endpoint backing click-to-select.

Ported from app/analytics/selected_config.py + the on_select handler in
app/analytics/performance.py (curve_number == 0, point_index → selected
trial). The endpoint returns an HTML fragment for one (metric, trial index)
pair: the trial's score for that metric, its delta vs the metric's best score
(no delta when the selected trial IS the best), and its hyperparameter config
table — the same content selected_config.py rendered, minus the click wiring
itself (that lives in the browser).

Uses the bundled test2.ihpo fixture (Random Search, 30 trials): trial 9
(index 8) is the best trial by accuracy (0.68125); trial 1 (index 0) scores
0.59375, clearly not the best.
"""

import pytest
from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment_with_result():
    from web.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _panel_url(pk, metric, idx):
    return reverse("web:trial_panel", args=[pk]) + f"?metric={metric}&idx={idx}"


def test_selecting_the_best_trial_shows_no_delta(client):
    """The best trial for a metric renders its score with no delta indicator."""
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 8))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Trial 9" in body
    assert "0.6813" in body or "0.68125" in body
    assert "metric-delta" not in body


def test_selecting_a_non_best_trial_shows_a_negative_delta(client):
    """A trial scoring below the metric's best shows a delta and its sign.

    Trial 1 scores 0.59375 vs the best 0.68125 → delta ≈ -0.0875 (Streamlit's
    selected_config.py: delta = selected.scores[metric] - best_score).
    """
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 0))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Trial 1" in body
    assert "0.5938" in body or "0.59375" in body
    assert "metric-delta" in body
    assert "down" in body  # rendered as a negative/"down" delta
    assert "0.0875" in body


def test_panel_includes_the_selected_trials_config(client):
    """The panel lists the selected trial's hyperparameters and values."""
    exp = _experiment_with_result()
    resp = client.get(_panel_url(exp.pk, "accuracy", 8))
    body = resp.content.decode()

    # test2.ihpo trials are Random-Forest configs with these hyperparameters
    for key in ("n_estimators", "max_depth", "min_samples_split", "max_features"):
        assert key in body


def test_panel_respects_the_requested_metric(client):
    """Selecting the same index under a different metric shows that metric's
    score and delta, not accuracy's."""
    exp = _experiment_with_result()
    resp_acc = client.get(_panel_url(exp.pk, "accuracy", 0))
    resp_f1 = client.get(_panel_url(exp.pk, "f1", 0))
    assert resp_acc.content != resp_f1.content


def test_rejects_unknown_metric(client):
    """A metric the experiment doesn't track is rejected, not a 500."""
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
    """An experiment with no result yet (never run) can't supply a trial panel."""
    from web.services import snapshot as adapter

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
