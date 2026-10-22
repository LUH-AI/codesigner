"""Proposals: the configuration cube's sampled configurations, picked off its
point cloud and selected in a trial's place, and run as a trial on request.

A proposal is one of the samples the cube's layers are drawn from — the same
sample, through the same cache (`ui.views.priors._slice_model`). Its panel shows what the
surrogate predicts for it rather than a score, and offers "Run as Trial": a run
of one, told to evaluate that configuration, recorded like any other run.

Runs here go through `execute_run` synchronously (see the autouse fixture in
tests/conftest.py), so a POST returns with the trial already recorded.
"""

import json

from django.urls import reverse

from core import io
from experiments.services import snapshot as adapter

from tests.conftest import DATASETS_DIR, export_ihpo, post_new_experiment


def _ran_experiment(client, name="proposals"):
    """A Random Forest on iris with a three-trial random search behind it, so
    there are trials to fit a surrogate to and a dataset to run against."""
    from experiments.models import Experiment
    from experiments.services.run import create_run, execute_run

    post_new_experiment(client, {
        "name": name, "model_name": "Random Forest",
        "optimizer_name": "Random Search", "seed": "7",
        "evaluation_scheme": "kfold", "evaluation_value": "3",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
    })
    exp = Experiment.objects.get(data__name=name)
    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()
    return exp


def _sample(exp, metric="accuracy"):
    from ui.views.priors import _cube_sample

    _built, _config_space, sample = _cube_sample(exp, metric)
    assert sample is not None
    return sample


def _panel(client, exp, n, metric="accuracy"):
    return client.get(reverse("ui:proposal_panel", args=[exp.pk]) + f"?metric={metric}&n={n}")


def test_a_proposal_panel_shows_a_prediction_and_offers_to_run_it(client):
    """Its number among the samples, the surrogate's prediction for it, its
    whole configuration, and the Run as Trial button — enabled, since nothing
    is running.

    Setup: a three-trial experiment; the sixth sample.
    Expect: "Proposal 6 of <n>", the word "predicted", an enabled Run as Trial
    posting that sample's index, and every value of its configuration.
    """
    exp = _ran_experiment(client)
    sample = _sample(exp)
    html = _panel(client, exp, 5).content.decode()

    assert f"Proposal 6 of {len(sample['configurations'])}" in html
    assert "predicted" in html
    form = html.split("data-run-proposal", 1)[1].split("</form>", 1)[0]
    assert 'name="n" value="5"' in form
    assert "Run as Trial" in form and "disabled" not in form
    for name in sample["configurations"][5]:
        assert f"<td>{name}</td>" in html


def test_a_proposal_that_is_not_in_the_sample_is_refused(client):
    exp = _ran_experiment(client)
    n = len(_sample(exp)["configurations"])

    assert _panel(client, exp, n).status_code == 400
    assert _panel(client, exp, "x").status_code == 400
    assert _panel(client, exp, 0, metric="not-a-metric").status_code == 400


def test_the_page_never_receives_the_exact_configurations(client):
    """The layer the page draws carries the samples rounded for the page; the
    exact configurations a proposal is run as stay on the server."""
    exp = _ran_experiment(client)
    data = client.get(reverse("ui:surrogate_prediction", args=[exp.pk]) + "?metric=accuracy").json()

    assert "configurations" not in data
    assert data["values"]


def test_running_a_proposal_makes_it_the_next_trial(client):
    """Run as Trial is a run of one, told which configuration to evaluate: the
    proposal comes back as the next trial, exactly as it was sampled, saying
    which proposal it was — and the optimized metric does not move.

    Setup: a three-trial experiment; post the eleventh sample.
    Expect: one new run, bounded at one trial, recording the proposal; a fourth
    trial with that exact configuration and origin "Proposal 11".
    """
    exp = _ran_experiment(client)
    config = _sample(exp)["configurations"][10]
    metric_before = exp.data.current_metric

    response = client.post(reverse("ui:run_proposal", args=[exp.pk]),
                           {"metric": "accuracy", "n": "10"})

    assert response.status_code == 302
    exp.refresh_from_db()
    run = exp.runs.order_by("-id").first()
    assert run is not None
    assert run.stopping == {"max_trials": 1}
    assert run.configurations[0]["proposal"] == 11
    assert run.configurations[0]["config"] == config
    assert exp.data.result is not None
    trials = exp.data.result["data"]
    assert len(trials) == 4
    stored = exp.data.result["configs"][str(trials[-1]["config_id"])]
    assert stored == config
    assert exp.data.current_metric == metric_before


def test_the_timeline_says_which_proposal_ran(client):
    """The run is recorded like any other, and says it ran a proposal rather
    than searched: its summary names the proposal and the trial it became, and
    its details list the configuration."""
    from ui import timeline

    exp = _ran_experiment(client)
    client.post(reverse("ui:run_proposal", args=[exp.pk]), {"metric": "accuracy", "n": "2"})
    exp.refresh_from_db()

    runs = [e for e in timeline.build(exp) if e["kind"] == "run"]
    newest = runs[0]
    assert "proposal 3" in newest["summary"] and "trial 4" in newest["summary"]
    assert any(d["label"] == "Proposal" for d in newest["details"])


def test_no_proposal_runs_while_a_run_is_under_way(client):
    """One run at a time: the panel's button is disabled and says why, and a
    post starts nothing.

    Setup: a three-trial experiment with a run marked as running.
    Expect: a disabled Run as Trial with the reason, and no new run after a
    post.
    """
    from experiments.models import Run

    exp = _ran_experiment(client)
    Run.objects.create(experiment=exp, stopping={"max_trials": 3},
                       primary_metric="accuracy", status="running")
    html = _panel(client, exp, 1).content.decode()
    form = html.split("data-run-proposal", 1)[1].split("</form>", 1)[0]
    runs_before = exp.runs.count()

    client.post(reverse("ui:run_proposal", args=[exp.pk]), {"metric": "accuracy", "n": "1"})

    assert "disabled" in form and "A run is under way." in html
    assert exp.runs.count() == runs_before


def test_a_proposal_run_survives_a_round_trip(client):
    """The file records what a run was told to evaluate, so an experiment
    reopened from it still knows which of its trials were proposals."""
    exp = _ran_experiment(client)
    client.post(reverse("ui:run_proposal", args=[exp.pk]), {"metric": "accuracy", "n": "4"})
    body = json.loads(export_ihpo(client, exp.pk).content)
    body["name"] = "reopened"
    copy = adapter.experiment_from_snapshot(io.parse(json.dumps(body).encode()))

    original = [r.configurations for r in exp.runs.order_by("id")]
    reopened = [r.configurations for r in copy.runs.order_by("id")]
    assert reopened == original
    assert reopened[-1][0]["proposal"] == 5


def test_the_page_knows_the_proposal_colour(client):
    """The cube rings a selected proposal in its own colour, sent by the
    server like every other selection colour so the page restates none."""
    from ui.figures import PROPOSAL_COLOR

    exp = _ran_experiment(client)
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    colours = json.loads(html.split('id="selection-colors-data"', 1)[1]
                         .split(">", 1)[1].split("</script>", 1)[0])

    assert colours["proposal"] == PROPOSAL_COLOR


def test_a_click_on_nothing_lets_go_of_a_proposal(client):
    """A proposal is a look rather than a place to stay: a click on nothing in
    particular — empty page, empty plot — goes back to the incumbent, where a
    trial would stay selected. Clicks on controls, on the proposal's own panel,
    and the ends of drags do not count. Asserted on the page script.
    """
    exp = _ran_experiment(client)
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    drop = html.split("function dropProposal(", 1)[1].split("\n            }\n", 1)[0]

    assert "proposalIdx[metric] === undefined" in drop
    assert "selectTrial(metric, bestIdxByMetric[metric])" in drop
    assert ".selected-config" in html.split("const CLICKABLE", 1)[1].split(";", 1)[0]
    listener = html.split("const CLICKABLE", 1)[1].split("// Everything that follows", 1)[0]
    assert "DRAG_SLOP" in listener and "dropProposal(currentMetric())" in listener
