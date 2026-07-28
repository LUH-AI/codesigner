"""The trials table shows a per-trial Duration column.

Duration comes from each trial's run_info["time"] (seconds), rendered to 3
decimals. Trials from files without timing render 0.000 s.
"""

import pytest
from django.urls import reverse

from ui.models import Experiment


def _experiment_with_timed_result():
    result = {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{
            "config_id": 1, "cost": 0.2, "time": 2.5, "cpu_time": 2.4,
            "starttime": 1000.0, "endtime": 1002.5, "status": 1,
            "seed": 0, "budget": None, "instance": None, "additional_info": {},
            "scores": {"accuracy": 0.8, "f1": 0.7, "precision": 0.75, "recall(macro)": 0.72},
            "incumbent_score": 0.8, "incumbent_config_id": 1,
        }],
        "configs": {"1": {"n_estimators": 100}}, "config_origins": {"1": "Random Search"},
        "optimizer_state": {}, "primary_metric": "accuracy", "best_score": 0.8,
        "best_config_id": "1", "hyperparameter_importance": {},
        "hyperparameter_importance_warning": {}, "trials_limit": None,
    }
    return Experiment.objects.create(
        name="dur", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1", "precision", "recall(macro)"],
        primary_metric="accuracy", original_metric="accuracy", seed=0, result=result,
    )


def test_trials_table_has_duration_column_and_value(client):
    exp = _experiment_with_timed_result()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "Duration" in body          # column header
    assert "2.500" in body             # the trial's duration, 3 decimals
