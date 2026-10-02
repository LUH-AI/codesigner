"""Forecasting with any model, and what a forecaster is told.

A forecast predicts a series' next *horizon* values from the values before
them — numbers for a regression, labels for a classification: forecasting is
how an experiment is evaluated, not a task of its own. Codesigner evaluates one the way it evaluates everything else — as folds
(`core.splits.backtest`), each training on the rows before an origin and
predicting the *horizon* time steps after it — so a forecaster is still a model
with `fit_predict(config, X_train, y_train, X_val)`. What it needs beyond that
is in `model.forecast`, set before its search space is read:

    {"horizon": H, "season": m, "gap": g,
     "time": column of X holding each row's time (as a number),
     "series": column of X naming each row's series, or None for one series,
     "known": the columns of X known ahead of time — the time's own parts and
              the series; every other column is only known once it happens}

The rows of `X_train` and `X_val` are in time order, so a series' history is
its training rows in the order given, and a validation row's step ahead is its
time's place among its series' validation times.

`Reduced` turns any model into a forecaster by *reduction*: each training
row becomes one example per step ahead, its features the series' values before
the forecast's origin — the last `forecast:lags` of them and a summary of the
last season — plus the step and the columns of the row known ahead of time:
its calendar and its series. Never its other columns, whose values in a future
row nobody has yet; trained on them, a forecast would be scored on information
it will not have when it is used. One model for every step (the step is a
feature) rather than one per step: a horizon of 24 would otherwise fit 24
models per trial.

For a regression the lags are the values, the summary is their mean, and the
model predicts the change from the origin's value, not the value, so a tree
can follow a trend it was never shown the far end of. For a classification
the lags are the labels' codes — categorical features — the summary is the
season's most frequent label, and the model predicts the label itself.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .columns import CATEGORICAL, DATETIME, NUMERIC
from .models.base import OutputBaseModel, Tunable

#: The prefix that marks a hyperparameter as the forecast's rather than the
#: wrapped regressor's.
PREFIX = "forecast:"

#: Season lengths by the step between a series' times.
SEASONS = {"h": 24, "D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "ME": 12,
           "Q": 4, "QS": 4, "QE": 4}


def infer_season(times) -> int:
    """How many steps make a season, from how far apart the times are: 12 for
    monthly data, 7 for daily, 24 for hourly, 4 for quarterly, 52 for weekly.
    1 — no season — when the spacing says nothing (yearly, irregular, numbers)."""
    stamps = pd.Series(pd.to_datetime(pd.Series(times), errors="coerce")).dropna()
    stamps = pd.DatetimeIndex(np.unique(stamps.to_numpy()))
    if len(stamps) < 3:
        return 1
    step = (stamps[1:] - stamps[:-1]).median()
    days = step / pd.Timedelta(days=1)
    for low, high, season in ((0.035, 0.05, 24), (0.9, 1.1, 7), (6.5, 7.5, 52),
                              (27, 32, 12), (88, 93, 4)):
        if low <= days <= high:
            return season
    return 1


def context(spec: dict, time_column: str, series_column: str, horizon: int,
            season: int, gap: int = 0) -> dict:
    """`model.forecast` for a dataset encoded by *spec* (`core.encoding`)."""
    from .encoding import feature_names

    names = feature_names(spec)
    kinds = {c["name"]: c["kind"] for c in spec["columns"]}
    time_feature = (f"{time_column}:days" if kinds.get(time_column) == DATETIME
                    else time_column)
    known = [i for i, name in enumerate(names)
             if name == time_column or name.startswith(f"{time_column}:")
             or (series_column and name == series_column)]
    return {"horizon": int(horizon), "season": max(1, int(season)), "gap": int(gap),
            "time": names.index(time_feature),
            "series": names.index(series_column) if series_column else None,
            "known": known}


def steps(X_train, X_val, forecast: dict):
    """Each validation row's series key and its step ahead of its series' last
    training row, counting the gap — 1 for the first time after the origin."""
    time, series = forecast["time"], forecast.get("series")
    key_train = _keys(X_train, series)
    key_val = _keys(X_val, series)
    ahead = np.zeros(len(X_val), dtype=int)
    for key in np.unique(key_val):
        rows = np.flatnonzero(key_val == key)
        times = X_val[rows, time]
        order = {t: i for i, t in enumerate(np.unique(times))}
        ahead[rows] = [order[t] + 1 + int(forecast.get("gap", 0)) for t in times]
    return key_train, key_val, ahead


def _keys(X, series):
    """Each row's series, as a string so a missing one is a series of its own."""
    if series is None:
        return np.zeros(len(X), dtype=object).astype(str)
    return np.array([str(v) for v in X[:, series]])


class Reduced(Tunable, OutputBaseModel):
    """A model, forecasting by reduction (see `core.forecasters.ReducedForecaster`).

    Built around a copy of the model it is given, for *task*. Its search space
    is the model's plus `forecast:lags`; its estimator learns the lag examples
    with the model, processed as the experiment says — which is why it takes
    no processing of its own columns, a series' times.
    """

    takes_processing = False
    #: The wrapped model's, set on each instance.
    name = "Reduced"

    def __init__(self, model, task: str = "regression"):
        self.model = copy.copy(model)
        self.model.task = task
        self.name = model.name
        self.tasks = (task,)
        self.task = task
        self.feature_kinds = tuple(getattr(model, "feature_kinds", ()))
        self.forecast = dict(getattr(model, "forecast", {}) or {})
        proba = getattr(model, "fit_predict_proba", None)
        self.probabilities = (task == "classification" and callable(proba)
                              and not getattr(proba, "_is_stub", False)
                              and bool(getattr(model, "probabilities", True)))
        self.dependencies = tuple(dict.fromkeys(("numpy",) + tuple(
            getattr(model, "dependencies", ()))))

    def get_config_space(self, seed: int = 0):
        from ConfigSpace import Integer

        cs = self.model.get_config_space(seed)
        season = self.forecast.get("season", 1)
        most = max(12, 2 * season)
        cs.add(Integer(f"{PREFIX}lags", (1, most), default=min(max(season, 3), most)))
        return cs

    def save_parameters(self, fitted, directory):
        """The model's parameters in its own format, the processing of the
        examples it learned from, and what a forecast is made from: each
        series' history (as codes, for labels) and the lags."""
        import json
        from pathlib import Path

        from .parameters import save_processing

        files = list(self.model.save_parameters(fitted.fitted_, directory))
        files += save_processing(fitted.processing_, directory)
        (Path(directory) / "series.json").write_text(json.dumps({
            "lags": fitted.lags,
            "labels": list(getattr(fitted, "labels_", {}) or {}),
            "history": {key: values.tolist() for key, values in fitted.history_.items()},
        }, indent=2))
        return files + ["series.json"]

    def parameter_howto(self):
        return (self.model.parameter_howto()
                + "\n# series.json: each series' history and the lags a forecast is "
                  "made from (see the exported model's ReducedForecaster).")

    def build(self, hyperparameters, data, seed=0):
        from .forecasters import ReducedForecaster
        from .processing import resolve

        self.model.feature_kinds = self.feature_kinds
        plan, _ = resolve(getattr(self.model, "processing", {}), self.model)
        return ReducedForecaster(
            model=self.model,
            hyperparameters={k: v for k, v in hyperparameters.items() if not k.startswith(PREFIX)},
            lags=int(hyperparameters.get(f"{PREFIX}lags", 3)),
            forecast=dict(self.forecast), plan=plan, task=self.task, seed=seed)
