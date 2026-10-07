"""Forecasting in `core`: the backtests, the metrics, and the forecasters.

Most of it runs on airline passengers (`datasets/airline.csv`), the classic
monthly series with a trend and a 12-month season, which every forecaster here
should beat repeating the last value on. Forecasting is a way of evaluating,
not a task: the same series is also forecast as labels.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from core import encoding, forecasting
from core.metrics import METRICS, metrics_for, score_all
from core.modelhost import launch_local, model_session
from core.models import ETSModel, SeasonalNaiveModel
from core.optimizers.trial import evaluate_trial
from core.registry import MODELS
from core.splits import backtest

from tests.conftest import DATASETS_DIR

from .test_modelhost import HEADER


def _airline(series=False, labels=False):
    """Airline as the run would see it: (splits, context, kinds). With
    *series*, two copies of it — the second doubled — as two series. With
    *labels*, the target is whether each month is busier than the one before."""
    frame = pd.read_csv(DATASETS_DIR / "airline.csv")
    if labels:
        frame["passengers"] = np.where(frame.passengers.diff() > 0, "busier", "quieter")
    if series:
        frame = pd.concat([frame.assign(route="A"),
                           frame.assign(route="B", passengers=frame.passengers * 2)])
        frame = frame[["month", "route", "passengers"]].sort_values("month", kind="stable")
    features = frame.iloc[:, :-1]
    spec = encoding.fit(features)
    times = pd.to_datetime(frame["month"]).to_numpy()
    season = forecasting.infer_season(frame["month"])
    y = frame.iloc[:, -1].to_numpy() if labels else frame.iloc[:, -1].to_numpy(float)
    splits = backtest(encoding.encode(spec, features), y, times,
                      horizon=12, folds=3, season=season,
                      series=frame["route"].to_numpy() if series else None)
    context = forecasting.context(spec, "month", "route" if series else "", 12, season)
    return splits, context, tuple(encoding.feature_kinds(spec))


def _scores(model, splits, context, kinds, config=None, task="regression"):
    model.feature_kinds, model.forecast = kinds, context
    config = config or dict(model.get_config_space().get_default_configuration())
    names = metrics_for(task, forecast=True, probabilities=task == "classification")
    scores, info = evaluate_trial(model, config, splits, {n: METRICS[n] for n in names}, seed=0)
    assert "error" not in (info.get("additional_info") or {}), info
    return scores


@pytest.mark.parametrize("dates, season", [
    (pd.date_range("2020-01-01", periods=30, freq="MS"), 12),
    (pd.date_range("2020-01-01", periods=30, freq="D"), 7),
    (pd.date_range("2020-01-01", periods=30, freq="h"), 24),
    (pd.date_range("2020-01-01", periods=30, freq="QS"), 4),
    (pd.date_range("2020-01-01", periods=30, freq="YS"), 1),
])
def test_the_season_is_read_from_how_far_apart_the_times_are(dates, season):
    """What: monthly is 12, daily 7, hourly 24, quarterly 4, yearly none.
    How: infers each from 30 regular timestamps."""
    assert forecasting.infer_season(dates.strftime("%Y-%m-%d %H:%M")) == season


def test_backtests_cut_every_series_at_the_same_origin():
    """What: each backtest validates the same next steps of every series and
    trains on all of their earlier rows, origins one horizon apart.
    How: two series of ten steps, horizon 2, three backtests, gap 1."""
    times = np.repeat(np.arange(10), 2)
    splits = backtest(np.c_[times], times.astype(float), times, horizon=2, folds=3, gap=1,
                      series=np.tile(["a", "b"], 10))

    assert [sorted(set(times[val])) for _, val in splits.folds] == [[4, 5], [6, 7], [8, 9]]
    assert [times[train].max() for train, _ in splits.folds] == [2, 4, 6]
    assert all(len(val) == 4 for _, val in splits.folds)


def test_mase_scales_by_each_series_own_seasonal_naive_error():
    """What: MASE is the error over the seasonal naive error of the history, so
    the same forecast of a series twice the size scores the same.
    How: scores one forecast against a 4-step season, then doubled."""
    history = {"y": np.array([10, 20, 30, 40, 12, 22, 32, 42.]), "series": None, "season": 4}
    one = METRICS["mase"].fn(np.array([14, 24.]), np.array([13, 25.]), history)
    doubled = METRICS["mase"].fn(np.array([28, 48.]), np.array([26, 50.]),
                                 {**history, "y": history["y"] * 2})

    assert one == pytest.approx(0.5) and doubled == pytest.approx(one)


def test_smape_is_a_symmetric_percentage():
    """What: sMAPE is 0 for a perfect forecast and treats over- and
    under-forecasting alike. How: three hand-checked cases."""
    smape = METRICS["smape"].fn
    assert smape([100.], [100.]) == 0
    assert smape([100.], [110.]) == pytest.approx(smape([110.], [100.]))
    assert smape([0.], [0.]) == 0


def test_a_metric_needing_history_without_it_says_so():
    """What: asking MASE to score without the history is a clear error.
    How: calls `score_all` without one."""
    with pytest.raises(ValueError, match="history"):
        score_all(np.array([1.]), np.array([1.]), {"mase": METRICS["mase"]})


@pytest.mark.parametrize("make", [ETSModel, SeasonalNaiveModel,
                                  lambda: forecasting.Reduced(MODELS["LightGBM"], "regression")],
                         ids=["ETS", "Seasonal Naive", "LightGBM reduced"])
def test_every_forecaster_beats_repeating_the_last_value(make):
    """What: on airline, each forecaster's error is below that of repeating
    the last value. How: backtests it and Seasonal Naive's "last" strategy."""
    splits, context, kinds = _airline()
    last = _scores(SeasonalNaiveModel(), splits, context, kinds, {"strategy": "last"})

    assert _scores(make(), splits, context, kinds)["rmse"] < last["rmse"]


def test_each_series_is_forecast_from_its_own_history():
    """What: with two series, a scale-free score is what each would get alone —
    the second series' history never leaks into the first's forecast.
    How: backtests ETS on airline and on airline plus a doubled copy."""
    alone = _scores(ETSModel(), *_airline())
    together = _scores(ETSModel(), *_airline(series=True))

    # Not exact: the smoother's optimiser lands a hair differently on the
    # doubled copy. A leak between the series would move it by far more.
    assert together["mase"] == pytest.approx(alone["mase"], rel=1e-4)


def test_the_reduction_adds_the_lags_to_the_models_space():
    """What: a reduced model tunes its own hyperparameters and `forecast:lags`,
    which defaults to a season, and keeps its task. How: reads the space."""
    model = forecasting.Reduced(MODELS["Random Forest"], "regression")
    model.forecast = {"season": 12}
    space = model.get_config_space()

    assert "n_estimators" in space and space["forecast:lags"].default_value == 12
    assert model.tasks == ("regression",) and model.model.task == "regression"


def test_forecast_metrics_are_offered_only_for_a_forecast():
    """What: MASE and sMAPE lead a regression forecast's metrics and are absent
    from any other regression; a classification forecast is scored like any
    classification. How: compares `metrics_for` with and without a forecast."""
    assert metrics_for("regression", forecast=True) == ["mase", "smape", "rmse", "mae", "r2"]
    assert metrics_for("regression") == ["rmse", "mae", "r2"]
    assert metrics_for("classification", forecast=True) == metrics_for("classification")


@pytest.mark.parametrize("name", ["Random Forest", "Logistic Regression"])
def test_a_classifier_forecasts_the_next_labels(name):
    """What: reduced for classification, a model forecasts each month's label
    from the labels before it and the calendar — well above always guessing
    the commoner label — and gives probabilities when the model does.
    How: backtests whether airline's months are busier than the one before."""
    model = forecasting.Reduced(MODELS[name], "classification")
    splits, context, kinds = _airline(labels=True)
    scores = _scores(model, splits, context, kinds, task="classification")

    labels = np.concatenate([splits.y[val] for _, val in splits.folds])
    commoner = max(np.mean(labels == "busier"), np.mean(labels == "quieter"))
    assert scores["accuracy"] > commoner + 0.1
    assert 0.5 < scores["roc_auc"] <= 1


def test_a_reduced_classifier_is_told_its_lags_are_labels():
    """What: the lagged labels and the season's commonest one are processed —
    and reach the model — as label columns, the step as a number, so a forest
    one-hots them. How: fits once and reads the kinds the examples' processing
    was given, and what it made of them."""
    model = forecasting.Reduced(MODELS["Random Forest"], "classification")
    splits, context, kinds = _airline(labels=True)
    model.feature_kinds, model.forecast = kinds, context
    train, _ = splits.folds[0]
    fitted = model.fit({**dict(model.get_config_space().get_default_configuration()),
                        "forecast:lags": 3}, splits.X[train], splits.y[train])

    assert fitted.processing_.kinds[:5] == ("numeric",) + ("categorical",) * 4
    assert fitted.processing_.plan["labels"] == "one_hot"


def test_a_forecaster_in_its_own_process_is_told_the_forecast(tmp_path):
    """What: `forecast` reaches a model in another process before its first
    trial. How: a model that writes what it was told to a file."""
    seen = tmp_path / "forecast.json"
    model_file = tmp_path / "model.py"
    model_file.write_text(HEADER + f'''

class Recorder(BaseModel):
    name = "Recorder"
    tasks = ("regression",)
    forecaster = True

    def get_config_space(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("k", (1, 5), default=3)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        with open({str(seen)!r}, "w") as f:
            f.write(__import__("json").dumps(self.forecast))
        return [float(y_train[-1])] * len(X_val)
''', encoding="utf-8")
    splits, context, _ = _airline()

    with model_session(launch_local(sys.executable, model_file), splits,
                       task="regression", forecast=context) as remote:
        evaluate_trial(remote, {"k": 1}, splits, {}, seed=0)

    assert json.loads(seen.read_text()) == context


def test_a_search_that_has_tried_everything_ends_instead_of_failing():
    """What: SMAC on Seasonal Naive's three strategies runs out of new
    configurations before its trial budget, and the run ends there, saying
    why, rather than raising. How: asks for ten trials of a three-point space."""
    from core.optimizers import SMACOptimizer
    from core.optimizers.base import STOPPED_BY_EXHAUSTED

    splits, context, kinds = _airline()
    model = SeasonalNaiveModel()
    model.feature_kinds, model.forecast = kinds, context
    result = SMACOptimizer().optimize(
        model, None, None, None, None, metrics={"mase": METRICS["mase"]},
        primary_metric="mase", n_trials=10, seed=0, splits=splits)

    assert 1 <= len(result.trials) <= 3
    assert result.metadata["stopped_by"] == STOPPED_BY_EXHAUSTED


@pytest.mark.parametrize("name", ["ETS", "Seasonal Naive", "LightGBM"])
def test_an_exported_forecaster_forecasts_what_comes_after_the_history(name, tmp_path):
    """What: exported with its forecast, a model is a script that reads the
    whole history and forecasts the next horizon of months — a regressor
    bringing the reduction with it — the same forecast it makes here.
    How: exports for airline, runs it on all 144 months in a fresh interpreter,
    and compares its 1961 forecast with fitting the model here."""
    import subprocess

    from core.model_export import export_model
    from core.processing import resolve

    frame = pd.read_csv(DATASETS_DIR / "airline.csv")
    spec = encoding.fit(frame.iloc[:, :-1])
    context = forecasting.context(spec, "month", "", 12, 12)
    registry_model = MODELS[name]
    reduce = not getattr(registry_model, "forecaster", False)
    model = (forecasting.Reduced(registry_model, "regression") if reduce
             else registry_model.__class__())
    model.feature_kinds, model.forecast = tuple(encoding.feature_kinds(spec)), context
    config = dict(model.get_config_space().get_default_configuration())
    plan, _ = resolve({}, getattr(model, "model", model))
    script = export_model(model, config=config, task="regression", seed=0, columns=spec,
                          plan=plan, about="A forecast.", forecast=context, time_column="month")
    path, out = tmp_path / "model.py", tmp_path / "forecast.csv"
    path.write_text(script)

    done = subprocess.run([sys.executable, str(path), str(DATASETS_DIR / "airline.csv"),
                           "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    forecast = pd.read_csv(out)

    assert forecast["month"].str[:7].tolist() == [f"1961-{m:02d}" for m in range(1, 13)]
    future = pd.DataFrame({"month": pd.date_range("1961-01-01", periods=12, freq="MS")})
    here = model.fit_predict(config, encoding.encode(spec, frame.iloc[:, :-1]),
                             frame.passengers.to_numpy(float), encoding.encode(spec, future), seed=0)
    np.testing.assert_allclose(forecast["passengers"], here, rtol=1e-6)


def test_a_reduced_forecast_never_reads_a_column_only_known_once_it_happens():
    """What: a column that is the target itself, sitting in the future rows,
    does not reach the regressor — only the time's parts and the series do —
    so a backtest scores what the forecast will know when it is used.
    How: adds a copy of the target as a column, then forecasts the last year
    once with that column's true future values and once with nonsense, and
    compares."""
    frame = pd.read_csv(DATASETS_DIR / "airline.csv")
    frame.insert(1, "copy", frame["passengers"])
    features = frame.iloc[:, :-1]
    spec = encoding.fit(features)
    X, y = encoding.encode(spec, features), frame["passengers"].to_numpy(float)
    model = forecasting.Reduced(MODELS["LightGBM"], "regression")
    model.feature_kinds = tuple(encoding.feature_kinds(spec))
    model.forecast = forecasting.context(spec, "month", "", 12, 12)
    config = dict(model.get_config_space().get_default_configuration())
    copy_column = encoding.feature_names(spec).index("copy")

    truthful = model.fit_predict(config, X[:-12], y[:-12], X[-12:], seed=0)
    nonsense = X[-12:].copy()
    nonsense[:, copy_column] = -1.0
    assert copy_column not in model.forecast["known"]
    np.testing.assert_array_equal(model.fit_predict(config, X[:-12], y[:-12], nonsense, seed=0),
                                  truthful)


def test_a_series_that_starts_inside_a_backtest_is_not_validated_in_it():
    """What: a series with no history before a backtest's origin has nothing to
    forecast from, so that backtest validates only the series that do — and
    every trial is still measured. How: airline plus a second route that
    exists only for the last six months, backtested with Seasonal Naive."""
    air = pd.read_csv(DATASETS_DIR / "airline.csv")
    frame = pd.concat([air.assign(route="A"), air.tail(6).assign(route="B")])
    frame = frame[["month", "route", "passengers"]].sort_values("month", kind="stable")
    spec = encoding.fit(frame.iloc[:, :-1])
    splits = backtest(encoding.encode(spec, frame.iloc[:, :-1]), frame.passengers.to_numpy(float),
                      pd.to_datetime(frame.month).to_numpy(), horizon=12, folds=2,
                      series=frame.route.to_numpy(), season=12)

    assert all(set(splits.series[val]) == {"A"} for _, val in splits.folds)
    scores = _scores(SeasonalNaiveModel(), splits,
                     forecasting.context(spec, "month", "route", 12, 12),
                     tuple(encoding.feature_kinds(spec)))
    assert np.isfinite(scores["mase"])


def test_an_exported_classification_forecast_forecasts_labels(tmp_path):
    """What: a classifier exported with its forecast reads a history of labels
    and forecasts the next horizon of them — the same labels it forecasts here.
    How: exports a reduced Random Forest for airline's busier/quieter months,
    runs it in a fresh interpreter, and compares with fitting it here."""
    import subprocess

    from core.model_export import export_model
    from core.processing import resolve

    frame = pd.read_csv(DATASETS_DIR / "airline.csv")
    frame["passengers"] = np.where(frame.passengers.diff() > 0, "busier", "quieter")
    history = tmp_path / "history.csv"
    frame.to_csv(history, index=False)
    spec = encoding.fit(frame.iloc[:, :-1])
    context = forecasting.context(spec, "month", "", 12, 12)
    registry_model = MODELS["Random Forest"]
    model = forecasting.Reduced(registry_model, "classification")
    model.feature_kinds, model.forecast = tuple(encoding.feature_kinds(spec)), context
    config = dict(model.get_config_space().get_default_configuration())
    plan, _ = resolve({}, model.model)
    script = export_model(model, config=config, task="classification", seed=0, columns=spec,
                          plan=plan, about="A forecast of labels.", forecast=context,
                          time_column="month")
    path, out = tmp_path / "model.py", tmp_path / "forecast.csv"
    path.write_text(script)

    done = subprocess.run([sys.executable, str(path), str(history), "--out", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr

    future = pd.DataFrame({"month": pd.date_range("1961-01-01", periods=12, freq="MS")})
    here = model.fit_predict(config, encoding.encode(spec, frame.iloc[:, :-1]),
                             frame.passengers.to_numpy(), encoding.encode(spec, future), seed=0)
    assert pd.read_csv(out)["passengers"].tolist() == list(here)
    assert set(here) <= {"busier", "quieter"}
