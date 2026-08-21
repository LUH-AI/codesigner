"""A trial has to carry every metric its experiment declares.

Every per-metric figure reads `trial.scores[metric]` for whichever metric the
page is showing, and the metric selector offers whatever `metrics.names` lists.
So a trial missing one of them is not a degraded chart — it is a KeyError at
render time, i.e. a 500 on a file that imported without complaint.

`io.parse` refuses it at the boundary instead, where the message can name the
trial and the metric. The accessors on `OptimizationResult` still guard, for
rows stored before that check existed; those are pinned in
`tests/core/test_derived_values.py`.
"""

import json

import pytest

from core import io

_SNAPSHOT = {
    "version": "0.1.0", "name": "scores", "model_name": "Random Forest",
    "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
    "primary_metric": "accuracy", "original_metric": "accuracy",
    "metric_names": ["accuracy", "f1"], "seed": 0, "dataset_path": "",
    "result": {
        "stats": {"submitted": 2, "finished": 2, "running": 0},
        "data": [
            {"config_id": 1, "cost": 0.2, "scores": {"accuracy": 0.8, "f1": 0.7},
             "incumbent_score": 0.8, "incumbent_config_id": 1},
            {"config_id": 2, "cost": 0.1, "scores": {"accuracy": 0.9, "f1": 0.85},
             "incumbent_score": 0.9, "incumbent_config_id": 2},
        ],
        "configs": {"1": {"n_estimators": 100}, "2": {"n_estimators": 200}},
        "config_origins": {"1": "Random Search", "2": "Random Search"},
        "optimizer_state": {}, "primary_metric": "accuracy",
        "best_score": 0.9, "best_config_id": "2", "trials_limit": None,
        "hyperparameter_importance": {}, "hyperparameter_importance_warning": {},
    },
}


def _parse(mutate=None):
    snapshot = json.loads(json.dumps(_SNAPSHOT))
    if mutate:
        mutate(snapshot)
    return io.parse(json.dumps(snapshot).encode())


def test_a_complete_result_parses():
    """The premise: this shape is accepted, so the failures below are about the
    missing score and not about the fixture."""
    assert len(_parse()["result"]["data"]) == 2


def test_a_trial_missing_a_declared_metric_is_refused():
    def drop_f1(snapshot):
        snapshot["result"]["data"][1]["scores"] = {"accuracy": 0.9}

    with pytest.raises(ValueError) as excinfo:
        _parse(drop_f1)

    message = str(excinfo.value)
    assert "trial 2" in message, "say which trial"
    assert "'f1'" in message, "say which metric"


def test_a_trial_with_no_scores_at_all_is_refused_when_more_than_one_metric():
    def drop_scores(snapshot):
        del snapshot["result"]["data"][0]["scores"]

    with pytest.raises(ValueError, match="no score"):
        _parse(drop_scores)


def test_a_single_metric_file_without_scores_still_parses():
    """The legacy shape `deserialize_result` already reads: an entry with no
    `scores` carries the primary metric alone, reconstructed from `cost`. That
    is complete exactly when the file declares no other metric, and refusing it
    would make every pre-multi-metric .ihpo unimportable."""
    def single_metric(snapshot):
        snapshot["metric_names"] = ["accuracy"]
        for entry in snapshot["result"]["data"]:
            del entry["scores"]

    assert len(_parse(single_metric)["result"]["data"]) == 2


def test_a_result_with_no_trials_is_unaffected():
    def empty(snapshot):
        snapshot["result"]["data"] = []

    assert _parse(empty)["result"]["data"] == []


def test_the_shipped_fixtures_still_parse():
    """Both fixtures predate this check; neither should be caught by it."""
    from tests.conftest import FIXTURES_DIR

    for name in ("test.ihpo", "test2.ihpo"):
        assert io.parse((FIXTURES_DIR / name).read_bytes())["result"]["data"]
