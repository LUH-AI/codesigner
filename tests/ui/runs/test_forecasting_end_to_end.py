"""A forecast, from the create form to a run and a file.

Airline passengers (`datasets/airline.csv`, monthly, 1949–1960) is forecast 12
months ahead from three rolling origins: a regression evaluated by backtests.
And the same months as labels, a classification evaluated the same way. Runs
execute inline in tests (see `runs_execute_synchronously`).
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
    """What: airline as a forecast stores its shape, infers a 12-month season,
    and a run measures every trial on the forecasting metrics — whether the
    model forecasts natively or is a regressor reduced to one.
    How: creates it, runs three trials on MASE, and reads the scores back."""
    from ui.models import Experiment
    from ui.views import _rebuild_result

    assert _create(client, model_name=model_name).status_code == 302
    exp = Experiment.objects.get(data__name="airline")
    assert (exp.data.task, exp.data.horizon, exp.data.cv_folds) == ("regression", 12, 3)
    assert exp.data.metric_names == ["mase", "smape", "rmse", "mae", "r2"]

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "mase"})
    exp.refresh_from_db()
    trials = _rebuild_result(exp).trials

    assert len(trials) == 3 and not any(t.failed for t in trials), [t.run_info for t in trials]
    assert all(0 < t.scores["mase"] < 5 and 0 < t.scores["smape"] < 60 for t in trials)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "3 backtests of 12 steps by month" in page


@pytest.mark.django_db
def test_the_forecast_survives_a_round_trip_through_a_file(client):
    """What: the file records the horizon, the season and the series column,
    and opening it again forecasts the same way. How: exports the snapshot and
    builds a second experiment from it."""
    from ui.models import Experiment
    from ui.services import snapshot as snapshot_adapter

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
    """What: no time column, more backtests than the history allows, or a
    series column the file lacks, each refused with the reason.
    How: posts the form with each and reads the page."""
    from ui.models import Experiment

    resp = _create(client, name="nope", **overrides)

    assert resp.status_code == 200 and says in resp.content.decode()
    assert not Experiment.objects.filter(data__name="nope").exists()


@pytest.mark.django_db
def test_a_model_that_only_forecasts_needs_backtests(client):
    """What: Seasonal Naive has nothing to predict from but a series' past, so
    an experiment validated any other way is refused for it.
    How: posts the form with cross-validation."""
    resp = _create(client, name="nope", model_name="Seasonal Naive",
                   evaluation_scheme="kfold", evaluation_value=3)

    assert resp.status_code == 200
    assert "Seasonal Naive only forecasts" in resp.content.decode()


@pytest.mark.django_db
def test_a_model_that_only_classifies_cannot_forecast_numbers(client):
    """What: backtests do not change what a model can predict: Logistic
    Regression still cannot regress. How: posts a regression forecast with it."""
    resp = _create(client, name="nope", model_name="Logistic Regression")

    assert resp.status_code == 200
    assert "Logistic Regression does not do regression" in resp.content.decode()


@pytest.mark.django_db
def test_an_upload_forecasts_only_if_it_says_so(client, settings):
    """What: an uploaded model runs in its own process, where nothing can make
    it a forecaster, so backtesting one is refused unless it declares
    `forecaster = True`. How: posts the same model file without and with it."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    from ui.models import Experiment

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
    """What: forecasting is a way of evaluating, not a task: whether each month
    is busier than the last is backtested like the passenger counts, scored on
    the classification metrics, the page says how, and the Forecast figure
    draws labels. How: uploads airline with a label target, creates a Random
    Forest classification under backtests, runs two trials, and fetches the
    figure."""
    import numpy as np
    import pandas as pd
    from django.core.files.uploadedfile import SimpleUploadedFile

    from ui.models import Experiment
    from ui.views import _rebuild_result

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
    trials = _rebuild_result(exp).trials

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
    """What: a forecast's Export model download is the forecasting script —
    its horizon and columns, and a regressor forecasting by reduction.
    How: runs one LightGBM trial and reads the downloaded file's constants."""
    import ast

    from ui.models import Experiment

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
    """What: a forecast's page has the Forecast figure, which draws the series
    and one forecast per backtest for the trial asked about; a classification
    experiment's page — validated any other way — has no such figure, and its
    endpoint refuses.
    How: runs one ETS trial, fetches the figure, then creates iris."""
    from ui.models import Experiment

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
