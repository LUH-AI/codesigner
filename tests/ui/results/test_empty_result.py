"""`io.parse` accepts a result with an empty `data` list from a hand-written or
truncated .ihpo, and the detail page must render it without a 500.
"""

import json

from core import io
from experiments.services import snapshot as adapter

_EMPTY_RESULT_SNAPSHOT = {
    "version": "0.1.0", "name": "empty-trials", "model_name": "Random Forest",
    "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
    "primary_metric": "accuracy", "original_metric": "accuracy",
    "metric_names": ["accuracy"], "seed": 0, "dataset_path": "",
    "result": {
        "stats": {}, "data": [], "configs": {}, "config_origins": {},
        "optimizer_state": {}, "primary_metric": "accuracy",
        "best_score": 0.0, "best_config_id": "0",
        "hyperparameter_importance": {}, "hyperparameter_importance_warning": {},
        "trials_limit": None,
    },
}


def _experiment_with_an_empty_result():
    return adapter.experiment_from_snapshot(
        io.parse(json.dumps(_EMPTY_RESULT_SNAPSHOT).encode()))


def test_parse_still_accepts_a_result_with_no_trials():
    parsed = io.parse(json.dumps(_EMPTY_RESULT_SNAPSHOT).encode())
    assert parsed["result"]["data"] == []


def test_the_detail_page_renders(client):
    exp = _experiment_with_an_empty_result()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert resp.status_code == 200


def test_it_renders_the_waiting_state_rather_than_empty_figures(client):
    """An empty result takes the "no result" path, so the template shows its
    waiting state."""
    exp = _experiment_with_an_empty_result()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert "panels" not in resp.context or not resp.context["panels"]


def test_the_figure_script_is_not_rendered_at_all(client):
    """`has_result` requires trials, so the plotting script stays off the page
    instead of breaking in the browser on `JSON.parse('""')`.
    """
    exp = _experiment_with_an_empty_result()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert 'class="prerun"' in body
    assert 'id="panels-data"' not in body, "the figure script must not render"
    assert 'id="metric-plots-data"' not in body
