"""The tests run mostly on airline passengers, a monthly series with a trend and
a 12-month season. Forecasting is a way of evaluating, and the same series is
also forecast as labels.
"""

import json
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest

from core import encoding, forecasting
from core.forecasters import ReducedForecaster
from core.metrics import METRICS, metrics_for, score_all
from core.modelhost import model_session
from core.models import ETSModel, SeasonalNaiveModel
from core.optimizers.trial import evaluate_trial
from core.registry import MODELS
from core.splits import backtest

from tests.conftest import DATASETS_DIR, launch_local

from .test_modelhost import HEADER


def _airline(series=False, labels=False):
    """`_airline` builds the run's (splits, context, kinds). The *series* flag
    adds a doubled second series, and the *labels* flag marks each month busier
    or quieter."""
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
    assert forecasting.infer_season(dates.strftime("%Y-%m-%d %H:%M")) == season


def test_backtests_cut_every_series_at_the_same_origin():
    """Backtest origins sit one horizon apart, and each backtest validates the
    same steps of every series."""
    times = np.repeat(np.arange(10), 2)
    splits = backtest(np.c_[times], times.astype(float), times, horizon=2, folds=3, gap=1,
                      series=np.tile(["a", "b"], 10))

    assert [sorted(set(times[val])) for _, val in splits.folds] == [[4, 5], [6, 7], [8, 9]]
    assert [times[train].max() for train, _ in splits.folds] == [2, 4, 6]
    assert all(len(val) == 4 for _, val in splits.folds)


def test_mase_scales_by_each_series_own_seasonal_naive_error():
    """MASE divides by the history's seasonal naive error, making the score
    scale-free."""
    history = {"y": np.array([10, 20, 30, 40, 12, 22, 32, 42.]), "series": None, "season": 4}
    one = METRICS["mase"].fn(np.array([14, 24.]), np.array([13, 25.]), history)
    doubled = METRICS["mase"].fn(np.array([28, 48.]), np.array([26, 50.]),
                                 {**history, "y": history["y"] * 2})

    assert one == pytest.approx(0.5) and doubled == pytest.approx(one)


def test_smape_is_a_symmetric_percentage():
    """`smape` is symmetric in forecast and truth."""
    smape = METRICS["smape"].fn
    assert smape([100.], [100.]) == 0
    assert smape([100.], [110.]) == pytest.approx(smape([110.], [100.]))
    assert smape([0.], [0.]) == 0


def test_a_metric_needing_history_without_it_says_so():
    with pytest.raises(ValueError, match="history"):
        score_all(np.array([1.]), np.array([1.]), {"mase": METRICS["mase"]})


@pytest.mark.parametrize("make", [ETSModel, SeasonalNaiveModel,
                                  lambda: forecasting.Reduced(MODELS["LightGBM"], "regression")],
                         ids=["ETS", "Seasonal Naive", "LightGBM reduced"])
def test_every_forecaster_beats_repeating_the_last_value(make):
    splits, context, kinds = _airline()
    last = _scores(SeasonalNaiveModel(), splits, context, kinds, {"strategy": "last"})

    assert _scores(make(), splits, context, kinds)["rmse"] < last["rmse"]


def test_each_series_is_forecast_from_its_own_history():
    """The second series' history must stay out of the first series' forecast."""
    alone = _scores(ETSModel(), *_airline())
    together = _scores(ETSModel(), *_airline(series=True))

    # A leak would move the score far more than the optimiser's jitter.
    assert together["mase"] == pytest.approx(alone["mase"], rel=1e-4)


def test_the_reduction_adds_the_lags_to_the_models_space():
    """`forecast:lags` defaults to one season."""
    model = forecasting.Reduced(MODELS["Random Forest"], "regression")
    model.forecast = {"season": 12}
    space = model.get_config_space()

    assert "n_estimators" in space and space["forecast:lags"].default_value == 12
    assert model.tasks == ("regression",) and model.model.task == "regression"


def test_forecast_metrics_are_offered_only_for_a_forecast():
    """`metrics_for` offers a classification forecast the usual classification
    metrics."""
    assert metrics_for("regression", forecast=True) == ["mase", "smape", "rmse", "mae", "r2"]
    assert metrics_for("regression") == ["rmse", "mae", "r2"]
    assert metrics_for("classification", forecast=True) == metrics_for("classification")


@pytest.mark.parametrize("name", ["Random Forest", "Logistic Regression"])
def test_a_classifier_forecasts_the_next_labels(name):
    """`forecasting.Reduced` predicts better than always guessing the commoner
    label."""
    model = forecasting.Reduced(MODELS[name], "classification")
    splits, context, kinds = _airline(labels=True)
    scores = _scores(model, splits, context, kinds, task="classification")

    labels = np.concatenate([splits.y[val] for _, val in splits.folds])
    commoner = max(np.mean(labels == "busier"), np.mean(labels == "quieter"))
    assert scores["accuracy"] > commoner + 0.1
    assert 0.5 < scores["roc_auc"] <= 1


def test_a_reduced_classifier_is_told_its_lags_are_labels():
    """The lagged labels arrive as label columns, and a forest one-hot encodes
    them."""
    model = forecasting.Reduced(MODELS["Random Forest"], "classification")
    splits, context, kinds = _airline(labels=True)
    model.feature_kinds, model.forecast = kinds, context
    train, _ = splits.folds[0]
    fitted = cast(ReducedForecaster, model.fit(
        {**dict(model.get_config_space().get_default_configuration()), "forecast:lags": 3},
        splits.X[train], splits.y[train]))

    assert fitted.processing_.kinds[:5] == ("numeric",) + ("categorical",) * 4
    assert fitted.processing_.plan["labels"] == "one_hot"


def test_a_forecaster_in_its_own_process_is_told_the_forecast(tmp_path):
    """The model in another process receives `forecast` before its first trial."""
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
    """SMAC exhausts a three-point space and ends the run with a reason instead
    of raising."""
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
    """The exported script predicts 1961 from all 144 months and matches a fit
    here."""
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
    """Future values of an unknown column must leave the forecast unchanged."""
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
    """A backtest only validates series with history before its origin."""
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
    """The exported classifier forecasts the same labels as fitting here."""
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
