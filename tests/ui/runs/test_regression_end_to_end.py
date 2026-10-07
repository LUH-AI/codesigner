"""A regression experiment, from the create form to an exported model.

Runs execute inline in tests (see `runs_execute_synchronously`), so posting
the Run form leaves a finished result behind.
"""

import ast

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, post_new_experiment


def _create(client, **overrides):
    data = {
        "name": "diabetes",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "diabetes.csv"), "task": "regression",
        "evaluation_scheme": "holdout",
        "evaluation_value": 0.2,
        "seed": 0,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


@pytest.mark.django_db
def test_a_regression_is_scored_on_the_regression_metrics(client):
    """What: diabetes created as a regression is stored as one and scored
    with the regression metrics. How: creates it and reads back what was
    stored."""
    from ui.models import Experiment

    assert _create(client).status_code == 302
    exp = Experiment.objects.get(data__name="diabetes")
    assert exp.data.task == "regression"
    assert exp.data.metric_names == ["rmse", "mae", "r2"]


@pytest.mark.django_db
def test_the_task_is_asked_never_guessed(client):
    """What: a form without a task is refused rather than guessed from the
    target, and a whole-number target is whatever the person says it is —
    wine's quality as classes, or as a quantity. How: posts diabetes with no
    task, then wine both ways."""
    from ui.models import Experiment

    resp = _create(client, name="unsaid", task="")
    assert resp.status_code == 200 and not Experiment.objects.filter(data__name="unsaid").exists()
    _create(client, name="wine classes", demo_dataset=str(DATASETS_DIR / "wine.csv"),
            task="classification")
    _create(client, name="wine quantity", demo_dataset=str(DATASETS_DIR / "wine.csv"))
    assert Experiment.objects.get(data__name="wine classes").data.task == "classification"
    assert Experiment.objects.get(data__name="wine quantity").data.task == "regression"


@pytest.mark.django_db
def test_a_text_target_cannot_be_regressed_on(client):
    """What: asking for regression on iris' text labels is refused, naming the
    column. How: posts the form and reads the error."""
    from ui.models import Experiment

    resp = _create(client, name="nope", demo_dataset=str(DATASETS_DIR / "iris.csv"),
                   task="regression")
    assert resp.status_code == 200
    assert "variety" in resp.content.decode()
    assert not Experiment.objects.filter(data__name="nope").exists()


@pytest.mark.django_db
def test_a_regression_run_finds_a_low_error_and_exports_it(client):
    """What: a regression run is scored and optimized in the right direction,
    the page draws it, and the best trial exports.
    How: runs six random-search trials on RMSE, checks the incumbent is the
    trial with the lowest RMSE, renders the page and downloads the model."""
    from ui.models import Experiment
    from ui.views import _rebuild_result

    _create(client)
    exp = Experiment.objects.get(data__name="diabetes")
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 6, "optimize_metric": "rmse"})

    result = _rebuild_result(exp)
    assert len(result.trials) == 6
    rmses = [t.scores["rmse"] for t in result.trials]
    assert all(0 < r < 200 for r in rmses)
    assert result.trials[result.best_index("rmse")].scores["rmse"] == min(rmses)

    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200

    body = client.get(reverse("ui:experiment_export_model", args=[exp.pk])).content.decode()
    ast.parse(body)
    assert "model.task = 'regression'" in body
    assert body.split('"""')[1].startswith("Random Forest predicting progression (regression).")
