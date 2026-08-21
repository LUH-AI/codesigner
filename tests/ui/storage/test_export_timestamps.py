"""Exporting asks whether the per-trial timestamps go in the file.

`starttime`/`endtime` say what time of day someone was working and on which
days — a fact about a person rather than about a search. The durations that make
the timing figures readable survive without them, so leaving them out costs the
file nothing it is for.

Asked at the point of export rather than kept as a setting, which is the change
these cover. A setting is answered once, by whoever set the instance up, for
every file it will ever send; the right answer depends on the file and on who is
about to receive it, neither of which is known when a settings page is being
filled in.
"""

import json

from django.urls import reverse

from ui.models import Experiment

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
    """A link to the export URL is a page, not a file: the question has to be
    put before the file exists, and it says what is at stake rather than naming
    a field."""
    exp = _experiment_with_timestamps()

    page = client.get(reverse("ui:experiment_export", args=[exp.pk]))
    body = page.content.decode()

    assert page.status_code == 200
    assert "Content-Disposition" not in page
    assert "what time of day you were working" in body
    assert 'value="strip"' in body and 'value="keep"' in body
    assert "Cancel" in body


def test_keeping_them_keeps_them(client):
    entry = _exported(client, _experiment_with_timestamps(), "keep")

    assert entry["starttime"] == 1700000000.0
    assert entry["endtime"] == 1700000002.0


def test_leaving_them_out_costs_the_file_nothing_else(client):
    """Which is why it is worth offering: every score, every configuration and
    how long each trial took are all still there."""
    entry = _exported(client, _experiment_with_timestamps(), "strip")

    assert "starttime" not in entry
    assert "endtime" not in entry
    assert entry["time"] == 2.0
    assert entry["scores"] == {"accuracy": 0.8}


def test_anything_but_keeping_them_strips_them(client):
    """The safe answer is the default one. A request that says nothing about
    timestamps — a stale bookmark, a script written against the old URL — gets
    the file without them rather than the file with."""
    exp = _experiment_with_timestamps()

    entry = json.loads(client.post(
        reverse("ui:experiment_export", args=[exp.pk])).content)["result"]["data"][0]

    assert "starttime" not in entry


def test_a_stripped_export_still_reimports(client):
    """Deserialize ignores the missing keys, so the file is a whole experiment
    either way."""
    from core import io

    body = export_ihpo(client, _experiment_with_timestamps().pk, "strip").content

    assert io.parse(body)["result"]["data"][0]["time"] == 2.0
