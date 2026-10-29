"""Airline passengers are forecast 12 months ahead from three backtest origins,
as a regression and as a classification.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, post_new_experiment

AIRLINE = DATASETS_DIR / "airline.csv"


def _create(client, name="airline", model_name="ETS", **overrides):
    data = {
        "name": name, "model_name": model_name, "optimizer_name": "Random Search",
        "demo_dataset": str(AIRLINE), "task": "regression", "time_column": "month",
        "horizon": 12, "evaluation_scheme": "backtest", "evaluation_value": 3, "seed": 0,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


@pytest.mark.django_db
@pytest.mark.parametrize("model_name", ["ETS", "LightGBM", "Seasonal Naive"])
def test_a_forecast_is_backtested_and_scored_on_mase(client, model_name):
    """Native forecasters and reduced regressors both score every trial on the
    forecasting metrics."""
    from experiments.models import Experiment
    from experiments.services.snapshot import rebuild_result

    assert _create(client, model_name=model_name).status_code == 302
    exp = Experiment.objects.get(data__name="airline")
    assert (exp.data.task, exp.data.horizon, exp.data.cv_folds) == ("regression", 12, 3)
    assert exp.data.metric_names == ["mase", "smape", "rmse", "mae", "r2"]

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "mase"})
    exp.refresh_from_db()
    result = rebuild_result(exp)
    assert result is not None
    trials = result.trials

    assert len(trials) == 3 and not any(t.failed for t in trials), [t.run_info for t in trials]
    assert all(0 < t.scores["mase"] < 5 and 0 < t.scores["smape"] < 60 for t in trials)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "3 backtests of 12 steps by month" in page


@pytest.mark.django_db
def test_the_forecast_survives_a_round_trip_through_a_file(client):
    """A reopened copy behaves the same, because the file keeps horizon, season
    and series column."""
    from experiments.models import Experiment
    from experiments.services import snapshot as snapshot_adapter

    _create(client, season=12)
    snapshot = snapshot_adapter.snapshot_from_experiment(Experiment.objects.get(data__name="airline"))
    assert snapshot["evaluation"]["horizon"] == 12 and snapshot["evaluation"]["season"] == 12

    snapshot["name"] = "reopened"
    copy = snapshot_adapter.experiment_from_snapshot(snapshot)
    assert (copy.data.task, copy.data.horizon, copy.data.season) == ("regression", 12, 12)
    assert snapshot["evaluation"]["scheme"] == "backtest"


@pytest.mark.django_db
@pytest.mark.parametrize("overrides, field, says", [
    ({"time_column": ""}, "time_column", "needs the column its rows are ordered"),
    ({"evaluation_value": 15}, "time_column", "too few for 15 backtests"),
    ({"series_column": "nowhere"}, "time_column", "no feature column named"),
])
def test_a_forecast_that_cannot_be_backtested_is_refused(client, overrides, field, says):
    """The form reports each unbacktestable setup as a validation error with
    its reason."""
    from experiments.models import Experiment

    resp = _create(client, name="nope", **overrides)

    assert resp.status_code == 200 and says in resp.content.decode()
    assert not Experiment.objects.filter(data__name="nope").exists()


@pytest.mark.django_db
def test_a_model_that_only_forecasts_needs_backtests(client):
    """The form reports a validation error for Seasonal Naive under any
    validation other than backtests, since Seasonal Naive needs a series'
    past."""
    resp = _create(client, name="nope", model_name="Seasonal Naive",
                   evaluation_scheme="kfold", evaluation_value=3)

    assert resp.status_code == 200
    assert "Seasonal Naive only forecasts" in resp.content.decode()


@pytest.mark.django_db
def test_a_model_that_only_classifies_cannot_forecast_numbers(client):
    resp = _create(client, name="nope", model_name="Logistic Regression")

    assert resp.status_code == 200
    assert "Logistic Regression does not do regression" in resp.content.decode()


@pytest.mark.django_db
def test_an_upload_forecasts_only_if_it_says_so(client, settings):
    """Backtests need the upload's `forecaster = True` declaration, because an
    upload runs in its own process."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    from experiments.models import Experiment

    settings.ALLOW_CUSTOM_MODELS = True
    source = (
        '# /// script\n# dependencies = ["ConfigSpace"]\n# ///\n'
        'from ConfigSpace import ConfigurationSpace, Integer\n'
        'from codesigner_model import BaseModel\n\n'
        'class Last(BaseModel):\n    name = "Last"\n    tasks = ("regression",)\n{flag}'
        '    def get_config_space(self, seed=0):\n'
        '        cs = ConfigurationSpace(seed=seed)\n'
        '        cs.add([Integer("k", (1, 3), default=1)])\n        return cs\n'
        '    def fit_predict(self, config, X_train, y_train, X_val, seed=0):\n'
        '        return [float(y_train[-1])] * len(X_val)\n')

    def create(name, flag):
        upload = SimpleUploadedFile("last.py", source.format(flag=flag).encode(),
                                    content_type="text/x-python")
        return _create(client, name=name, model_name="", model_file=upload)

    refused = create("plain", "")
    assert refused.status_code == 200
    assert "does not forecast" in refused.content.decode()
    assert create("declared", "    forecaster = True\n").status_code == 302
    assert Experiment.objects.get(data__name="declared").data.horizon == 12


@pytest.mark.django_db
def test_a_classification_is_forecast_too(client):
    """Backtesting is an evaluation scheme, so a label target forecasts too,
    scored on classification metrics with a label-axis figure."""
    import numpy as np
    import pandas as pd
    from django.core.files.uploadedfile import SimpleUploadedFile

    from experiments.models import Experiment
    from experiments.services.snapshot import rebuild_result

    frame = pd.read_csv(AIRLINE)
    frame["busier"] = np.where(frame.pop("passengers").diff() > 0, "yes", "no")
    upload = SimpleUploadedFile("busier.csv", frame.to_csv(index=False).encode())
    resp = _create(client, name="busier", model_name="Random Forest", task="classification",
                   demo_dataset="", dataset_file=upload)
    assert resp.status_code == 302, resp.content.decode()[-2000:]
    exp = Experiment.objects.get(data__name="busier")
    assert (exp.data.task, exp.data.horizon) == ("classification", 12)
    assert "accuracy" in exp.data.metric_names and "mase" not in exp.data.metric_names

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 2, "optimize_metric": "accuracy"})
    exp.refresh_from_db()
    result = rebuild_result(exp)
    assert result is not None
    trials = result.trials

    assert len(trials) == 2 and not any(t.failed for t in trials), [t.run_info for t in trials]
    assert all(0.5 < t.scores["accuracy"] <= 1 for t in trials)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "3 backtests of 12 steps by month" in page and 'id="figure-forecast"' in page
    figure = client.get(reverse("ui:forecast", args=[exp.pk]),
                        {"metric": "accuracy", "idx": 0}).json()["figure"]
    assert figure["layout"]["yaxis"]["type"] == "category"
    assert all(set(trace["y"]) <= {"yes", "no"} for trace in figure["data"])


@pytest.mark.django_db
def test_the_exported_forecaster_carries_its_forecast(client):
    """The download is the forecasting script, reducing a regressor to a
    forecaster."""
    import ast

    from experiments.models import Experiment

    _create(client, model_name="LightGBM")
    exp = Experiment.objects.get(data__name="airline")
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 1, "optimize_metric": "mase"})

    body = client.get(reverse("ui:experiment_export_model", args=[exp.pk])).content.decode()
    tree = ast.parse(body)
    assigned = {n.targets[0].id: n.value for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}

    assert ast.literal_eval(assigned["FORECAST"])["horizon"] == 12
    assert ast.literal_eval(assigned["TIME_COLUMN"]) == "month"
    assert "class ReducedForecaster" in body and "class LightGBM(OutputBaseModel)" in body
    assert "--horizon" in body


@pytest.mark.django_db
def test_the_forecast_figure_draws_every_backtest_and_only_for_a_forecast(client):
    """Only a forecast gets the figure, with one trace per backtest."""
    from experiments.models import Experiment

    _create(client)
    exp = Experiment.objects.get(data__name="airline")
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 1, "optimize_metric": "mase"})

    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="figure-forecast"' in page
    data = client.get(reverse("ui:forecast", args=[exp.pk]), {"metric": "mase", "idx": 0}).json()
    assert data["warning"] is None
    assert [t["name"] for t in data["figure"]["data"]][1:] == [f"Backtest {n}" for n in (1, 2, 3)]
    assert all(len(t["x"]) == 12 for t in data["figure"]["data"][1:])

    post_new_experiment(client, {
        "name": "iris", "model_name": "Random Forest", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification", "seed": 0})
    iris = Experiment.objects.get(data__name="iris")
    assert 'id="figure-forecast"' not in client.get(
        reverse("ui:experiment_detail", args=[iris.pk])).content.decode()
    assert client.get(reverse("ui:forecast", args=[iris.pk]),
                      {"metric": "accuracy", "idx": 0}).status_code == 400
