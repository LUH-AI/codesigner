"""A regression experiment, from the create form to an exported model.

Runs execute inline in tests (see `runs_execute_synchronously`), so posting
the Run form leaves a finished result behind.
"""

import ast

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR


def _create(client, **overrides):
    data = {
        "name": "diabetes",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "diabetes.csv"),
        "evaluation_scheme": "holdout",
        "evaluation_value": 0.2,
        "seed": 0,
    }
    data.update(overrides)
    return client.post(reverse("ui:new_experiment"), data)


@pytest.mark.django_db
def test_the_task_is_guessed_from_the_target(client):
    """What: left on guess, diabetes' numeric target makes a regression
    experiment scored with the regression metrics.
    How: creates one and reads back what was stored."""
    from ui.models import Experiment

    assert _create(client).status_code == 302
    exp = Experiment.objects.get(data__name="diabetes")
    assert exp.data.task == "regression"
    assert exp.data.metric_names == ["rmse", "mae", "r2"]


@pytest.mark.django_db
def test_asking_for_classification_overrides_the_guess(client):
    """What: the guess is only a default. How: creates iris as classification
    explicitly, and wine's integer target likewise stays classification."""
    from ui.models import Experiment

    _create(client, name="iris", demo_dataset=str(DATASETS_DIR / "iris.csv"),
            task="classification")
    assert Experiment.objects.get(data__name="iris").data.task == "classification"
    _create(client, name="wine", demo_dataset=str(DATASETS_DIR / "wine.csv"))
    assert Experiment.objects.get(data__name="wine").data.task == "classification"


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
    tree = ast.parse(body)
    task = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "TASK")
    assert ast.literal_eval(task.value) == "regression"
