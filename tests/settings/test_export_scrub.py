"""Exporting an .ihpo scrubs absolute timestamps when the setting is off.

`export_absolute_times=False` strips `starttime`/`endtime` (the "when it ran"
info) from every trial entry, keeping `time` (the duration). Reconstruction and
re-import are unaffected (deserialize ignores those keys).
"""

import json

import pytest
from django.urls import reverse

from web.models import Experiment


def _experiment_with_timestamps(**kw):
    result = {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.2, "time": 2.0,
                  "starttime": 1700000000.0, "endtime": 1700000002.0, "cpu_time": 1.9,
                  "status": 1, "seed": 0, "budget": None, "instance": None, "additional_info": {},
                  "scores": {"accuracy": 0.8}, "incumbent_score": 0.8, "incumbent_config_id": 1}],
        "configs": {"1": {"n_estimators": 100}}, "config_origins": {"1": "Random Search"},
        "optimizer_state": {}, "primary_metric": "accuracy", "best_score": 0.8,
        "best_config_id": "1", "hyperparameter_importance": {},
        "hyperparameter_importance_warning": {}, "trials_limit": None,
    }
    base = dict(name="exp", model_name="Random Forest", optimizer_name="Random Search",
                metric_names=["accuracy"], primary_metric="accuracy", original_metric="accuracy",
                seed=0, result=result)
    base.update(kw)
    return Experiment.objects.create(**base)


def _exported(client, exp):
    resp = client.get(reverse("web:experiment_export", args=[exp.pk]))
    return json.loads(resp.content.decode())["result"]["data"][0]


def test_export_keeps_timestamps_by_default(client):
    exp = _experiment_with_timestamps()  # inherits default True
    entry = _exported(client, exp)
    assert entry["starttime"] == 1700000000.0
    assert entry["endtime"] == 1700000002.0


def test_export_scrubs_timestamps_when_off(client):
    exp = _experiment_with_timestamps(
        use_default_settings=False, settings={"export_absolute_times": False})
    entry = _exported(client, exp)
    assert "starttime" not in entry
    assert "endtime" not in entry
    assert entry["time"] == 2.0        # duration kept
    assert entry["scores"] == {"accuracy": 0.8}


def test_scrubbed_export_still_reimports(client):
    """A scrubbed .ihpo re-imports fine (deserialize ignores the missing keys)."""
    from core import io
    exp = _experiment_with_timestamps(
        use_default_settings=False, settings={"export_absolute_times": False})
    body = client.get(reverse("web:experiment_export", args=[exp.pk])).content
    snapshot = io.parse(body)
    assert snapshot["result"]["data"][0]["time"] == 2.0
