"""Forecasting is an evaluation mode for regression or classification, and
`core.splits.backtest` scores it as folds, so a forecaster is still a model with
`fit_predict`. `model.forecast` (see `context`) carries the horizon, season,
gap, time, series and known-ahead columns, and is set before the search space is
read. `Reduced` turns any model into a forecaster from lags and known-ahead
columns only, since other columns have no future values to forecast from. One
model learns every step, since a horizon of 24 would otherwise fit 24 models per
trial.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from codesigner_model import OutputBaseModel

from .columns import DATETIME
from .models.base import Tunable

#: Marks a hyperparameter as the forecast's own.
PREFIX = "forecast:"

def infer_season(times) -> int:
    """`infer_season` returns 12 for monthly data, 7 for daily, 24 for hourly, 4
    for quarterly, 52 for weekly, and 1 otherwise."""
    stamps = pd.Series(pd.to_datetime(pd.Series(times), errors="coerce")).dropna()
    stamps = pd.DatetimeIndex(np.unique(stamps.to_numpy()))
    if len(stamps) < 3:
        return 1
    step = (stamps[1:] - stamps[:-1]).median()  # pyright: ignore[reportAttributeAccessIssue]
    days = step / pd.Timedelta(days=1)
    for low, high, season in ((0.035, 0.05, 24), (0.9, 1.1, 7), (6.5, 7.5, 52),
                              (27, 32, 12), (88, 93, 4)):
        if low <= days <= high:
            return season
    return 1


def context(spec: dict, time_column: str, series_column: str, horizon: int,
            season: int, gap: int = 0) -> dict:
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


class Reduced(Tunable, OutputBaseModel):
    """`Reduced` wraps a copy of the model and adds `forecast:lags` to the
    model's space. `Reduced` takes no processing of its own, since the estimator
    processes the lag examples as the experiment says.
    """

    takes_processing = False
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
        """`save_parameters` also writes ``series.json``, with each series'
        history (as codes, for labels) and the lags."""
        import json
        from pathlib import Path

        from .fitted_model import save_processing

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


def backtests(model, config, splits, times, rows, seed, labels=False):
    """`backtests` returns each fold's `(times, predictions)` on *rows*."""
    out = []
    for train_idx, val_idx in splits.folds:
        predicted = np.asarray(model.fit_predict(config, splits.X[train_idx],
                                                 splits.y[train_idx], splits.X[val_idx],
                                                 seed=seed),
                               dtype=object if labels else float)
        mine = np.isin(val_idx, rows)
        out.append((times[val_idx[mine]], predicted[mine]))
    return out
