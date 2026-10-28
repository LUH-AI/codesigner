"""Before any trial, `experiment_prerun.html` stands in for the figures and
keeps the optimizer editable."""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, post_new_experiment
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db


def _created(client):
    post_new_experiment(client, {
        "name": "before", "task": "classification", "model_name": "Random Forest",
        "optimizer_name": "Random Search", "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "evaluation_scheme": "holdout", "evaluation_value": 0.2, "seed": 0})
    return Experiment.objects.get(data__name="before")


def _trials(exp):
    exp.data.result = {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.5, "time": 1.0, "scores": {"accuracy": 0.5},
                  "incumbent_config_id": 1}],
        "configs": {"1": {"max_depth": 5}}, "config_origins": {}, "optimizer_state": {},
        "primary_metric": "accuracy", "best_score": 0.5, "best_config_id": "1",
    }
    exp.data.save()


def test_a_new_experiment_opens_on_its_own_page(client):
    exp = _created(client)
    response = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    html = response.content.decode()

    assert [t.name for t in response.templates][0] == "ui/experiment_prerun.html"
    assert f'action="{reverse("ui:experiment_optimizer", args=[exp.pk])}"' in html
    assert 'name="optimizer_name"' in html and "fig-tabbar" not in html
    views = html.split('experiment-views">', 1)[1].split("</div>", 1)[0]
    assert 'aria-current="page">Dashboard</a>' in views


def test_the_optimizer_can_change_before_the_first_trial(client):
    """The change lands on the timeline as `optimizer_changed`."""
    exp = _created(client)
    client.post(reverse("ui:experiment_optimizer", args=[exp.pk]), {"optimizer_name": "SMAC"})
    exp.refresh_from_db()
    timeline = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()

    assert exp.data.optimizer_name == "SMAC"
    assert exp.data.history.filter(kind="optimizer_changed").count() == 1
    assert "Optimizer changed: Random Search → SMAC." in timeline


def test_once_there_are_trials_the_dashboard_takes_over(client):
    """Once a trial exists, the next run continues that search, so the optimizer
    stays fixed."""
    exp = _created(client)
    _trials(exp)
    response = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    client.post(reverse("ui:experiment_optimizer", args=[exp.pk]), {"optimizer_name": "SMAC"})
    exp.refresh_from_db()

    assert [t.name for t in response.templates][0] == "ui/experiment_detail.html"
    assert exp.data.optimizer_name == "Random Search"


def test_a_run_under_way_locks_the_optimizer(client):
    """The page stays up with the optimizer disabled and a reason given."""
    exp = _created(client)
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="running",
                       stopping={"max_trials": 3}, trial_offset=0, trial_count=0)
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    form = html.split('class="card prerun-optimizer"', 1)[1].split("</section>", 1)[0]

    assert '<select id="optimizer_name" name="optimizer_name" disabled>' in form
    assert "A run is under way with this optimizer." in form
    assert 'type="submit"' not in form
