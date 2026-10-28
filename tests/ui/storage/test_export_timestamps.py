"""`starttime` and `endtime` reveal when a person worked, and the durations
survive without them. Export asks per file, because the right answer depends on
the file and its recipient.
"""

import json
import re

from django.urls import reverse

from experiments.models import Experiment

from tests.conftest import export_ihpo


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
                metric_names=["accuracy"], current_metric="accuracy", original_metric="accuracy",
                seed=0, result=result)
    base.update(kw)
    return Experiment.objects.create(**base)


def _exported(client, exp, timestamps):
    return json.loads(
        export_ihpo(client, exp.pk, timestamps).content.decode())["result"]["data"][0]


def test_asking_comes_before_downloading(client):
    """The question comes before the file exists and names what is at stake."""
    exp = _experiment_with_timestamps()

    page = client.get(reverse("ui:experiment_export", args=[exp.pk]))
    body = page.content.decode()

    assert page.status_code == 200
    assert "Content-Disposition" not in page
    assert "Real times show when you were working." in body
    # Relative times are preselected, and a single Export button serves two
    # independent questions.
    assert re.search(r'name="timestamps" value="relative" checked', body)
    assert 'name="timestamps" value="absolute"' in body
    assert not re.search(r'name="tracebacks"[^>]*checked', body)
    assert "Cancel" in body


def test_keeping_them_keeps_them(client):
    entry = _exported(client, _experiment_with_timestamps(), "keep")

    assert entry["starttime"] == 1700000000.0
    assert entry["endtime"] == 1700000002.0


def test_leaving_them_out_costs_the_file_nothing_else(client):
    entry = _exported(client, _experiment_with_timestamps(), "strip")

    assert "starttime" not in entry
    assert "endtime" not in entry
    assert entry["time"] == 2.0
    assert entry["scores"] == {"accuracy": 0.8}


def test_anything_but_keeping_them_strips_them(client):
    """A request silent on timestamps, such as a stale bookmark, gets the file
    without them."""
    exp = _experiment_with_timestamps()

    entry = json.loads(client.post(
        reverse("ui:experiment_export", args=[exp.pk])).content)["result"]["data"][0]

    assert "starttime" not in entry


def test_a_stripped_export_still_reimports(client):
    """Deserialize ignores the missing keys."""
    from core import io

    body = export_ihpo(client, _experiment_with_timestamps().pk, "strip").content

    assert io.parse(body)["result"]["data"][0]["time"] == 2.0
