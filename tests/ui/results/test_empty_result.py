"""A result that carries no trials must not break the detail page.

`io.parse` accepts a snapshot whose result has an empty `data` list, so this is
reachable by importing a hand-written or truncated .ihpo. A run never produces
one — `ui/services/run.py` only writes a result that has trials — which is why
it went unnoticed until the derived-values consolidation walked past it.

Before the fix this raised (an argmax over an empty range) and returned a 500.
"""

import json

from core import io
from ui.services import snapshot as adapter

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
    """Pinning the premise: this is a real shape reaching the app, not one the
    parser was already rejecting."""
    parsed = io.parse(json.dumps(_EMPTY_RESULT_SNAPSHOT).encode())
    assert parsed["result"]["data"] == []


def test_the_detail_page_renders(client):
    exp = _experiment_with_an_empty_result()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert resp.status_code == 200


def test_it_renders_the_waiting_state_rather_than_empty_figures(client):
    """It takes the same path as "no result at all": no panels, so the template
    shows its waiting state instead of drawing figures over nothing."""
    exp = _experiment_with_an_empty_result()
    resp = client.get(f"/experiments/{exp.pk}/")
    assert "panels" not in resp.context or not resp.context["panels"]


def test_the_figure_script_is_not_rendered_at_all(client):
    """The gap the status-code check above missed.

    Returning 200 was not enough: `has_result` was `result is not None`, so an
    empty-trials result still rendered the whole plotting script while the early
    return meant `panels`/`metric_plots` were absent from the context. Django
    resolves the missing variables to `""`, so the page loaded and then broke in
    the browser on `JSON.parse('""').forEach`. `has_result` now means "a result
    with trials", which is what the early return actually keys on.
    """
    exp = _experiment_with_an_empty_result()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert "No results yet." in body
    assert 'id="panels-data"' not in body, "the figure script must not render"
    assert 'id="metric-plots-data"' not in body
