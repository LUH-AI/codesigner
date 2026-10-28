"""Each forecaster is a scikit-learn estimator that learns a series' past in
`fit` and forecasts the asked rows in `predict`, per `forecast`
(`core.forecasting.context`). An exported model can include `core.forecasters`
because `core.forecasters` imports only `codesigner_model` and `.processing`.
"""

import warnings

import numpy as np
from codesigner_model import OutputBaseModel
from sklearn.base import BaseEstimator

from .processing import Processing

NUMERIC = "numeric"
CATEGORICAL = "categorical"


def series_keys(X, series):
    """A missing series becomes a series of its own."""
    if series is None:
        return np.zeros(len(X), dtype=object).astype(str)
    return np.array([str(v) for v in np.asarray(X)[:, series]])


def steps_ahead(X, forecast):
    """The first forecast time is step 1 + gap."""
    X = np.asarray(X, dtype=float)
    keys = series_keys(X, forecast.get("series"))
    ahead = np.zeros(len(X), dtype=int)
    time, gap = forecast.get("time", 0), int(forecast.get("gap", 0))
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        order = {t: i for i, t in enumerate(np.unique(X[rows, time]))}
        ahead[rows] = [order[t] + 1 + gap for t in X[rows, time]]
    return keys, ahead


class _Estimator(BaseEstimator):

    forecast: dict

    def _history(self, X, y):
        keys = series_keys(X, self.forecast.get("series"))
        y = np.asarray(y, dtype=float)
        return {key: y[keys == key] for key in np.unique(keys)}


class SeasonalNaiveForecaster(_Estimator):
    """``"seasonal"`` repeats the last season, ``"last"`` repeats the last
    value, and ``"drift"`` extends the first-to-last line."""


    def __init__(self, strategy="seasonal", forecast=None):
        self.strategy = strategy
        self.forecast = forecast or {}

    def fit(self, X, y):
        self.history_ = self._history(X, y)
        return self

    def predict(self, X):
        season = max(1, int(self.forecast.get("season", 1)))
        keys, ahead = steps_ahead(X, self.forecast)
        out = np.full(len(keys), np.nan)
        for j, (key, step) in enumerate(zip(keys, ahead)):
            values = self.history_.get(key, np.array([]))
            if not len(values):
                continue
            if self.strategy == "seasonal" and len(values) >= season:
                out[j] = values[len(values) - season + (step - 1) % season]
            elif self.strategy == "drift" and len(values) > 1:
                out[j] = values[-1] + step * (values[-1] - values[0]) / (len(values) - 1)
            else:
                out[j] = values[-1]
        return out


class ETSForecaster(_Estimator):
    """`ETSForecaster` fits one statsmodels Holt-Winters smoother per series.
    `ETSForecaster` drops the season for a series shorter than two seasons and
    swaps ``"mul"`` to ``"add"`` for a series with a non-positive value.
    """


    def __init__(self, trend="add", damped=False, seasonal="add", forecast=None):
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.forecast = forecast or {}

    def fit(self, X, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        season = max(1, int(self.forecast.get("season", 1)))
        self.fits_ = {}
        for key, values in self._history(X, y).items():
            trend = None if self.trend == "none" else "add"
            seasonal = None if self.seasonal == "none" or season < 2 else self.seasonal
            if seasonal and len(values) < 2 * season:
                seasonal = None
            if seasonal == "mul" and np.any(values <= 0):
                seasonal = "add"
            with warnings.catch_warnings():
                # Convergence and frequency notices are unactionable here.
                warnings.simplefilter("ignore")
                self.fits_[key] = ExponentialSmoothing(
                    values, trend=trend, damped_trend=bool(trend) and bool(self.damped),
                    seasonal=seasonal, seasonal_periods=season if seasonal else None,
                    initialization_method="estimated").fit()
        return self

    def predict(self, X):
        keys, ahead = steps_ahead(X, self.forecast)
        out = np.full(len(keys), np.nan)
        for key in np.unique(keys):
            if key not in self.fits_:
                continue
            rows = np.flatnonzero(keys == key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = self.fits_[key].forecast(int(ahead[rows].max()))
            out[rows] = path[ahead[rows] - 1]
        return out


class ReducedForecaster(_Estimator):
    """Each training row becomes one example per step ahead: the step, the last
    *lags* values before the origin, the last season's mean or commonest label,
    and the row's known columns. One model learns every step and, for numbers,
    predicts the change from the origin's value, so a tree can follow a trend.
    """


    def __init__(self, model=None, hyperparameters=None, lags=3, forecast=None, plan=None,
                 task="regression", seed=0):
        self.model = model
        self.hyperparameters = hyperparameters or {}
        self.lags = lags
        self.forecast = forecast or {}
        self.plan = plan
        self.task = task
        self.seed = seed

    @property
    def classifies(self):
        return self.task == "classification"

    def _known(self):
        known = self.forecast.get("known")
        return list(known) if known is not None else [self.forecast.get("time", 0)]

    def _features(self, before, step, row):
        recent = before[::-1][:self.lags]
        padded = np.full(self.lags, np.nan)
        padded[:len(recent)] = recent
        season = before[-max(1, int(self.forecast.get("season", 1))):]
        season = season[~np.isnan(season)]
        if not len(season):
            level = np.nan
        elif self.classifies:
            codes, counts = np.unique(season, return_counts=True)
            level = codes[np.argmax(counts)]
        else:
            level = season.mean()
        own = np.asarray(row, dtype=float)[self._known()]
        return np.concatenate([[step], padded, [level], own])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        keys = series_keys(X, self.forecast.get("series"))
        if self.classifies:
            # Lagged labels become codes by training-label order.
            self.labels_ = {label: float(i) for i, label in enumerate(np.unique(y.astype(str)))}
            values = np.array([self.labels_[str(v)] for v in y], dtype=float)
        else:
            values = y.astype(float)
        self.history_ = {key: values[keys == key] for key in np.unique(keys)}

        gap = int(self.forecast.get("gap", 0))
        top = int(self.forecast.get("horizon", 1)) + gap
        rows, targets = [], []
        for key in self.history_:
            coded, raw, exog = values[keys == key], y[keys == key], X[keys == key]
            for position in range(len(coded)):
                for step in range(1 + gap, top + 1):
                    origin = position - step
                    if origin < 0:
                        break
                    rows.append(self._features(coded[:origin + 1], step, exog[position]))
                    targets.append(raw[position] if self.classifies
                                   else coded[position] - coded[origin])
        if not rows:
            raise ValueError("no series has enough history to learn a forecast from")

        kinds = tuple(getattr(self.model, "feature_kinds", ()) or ())
        lag_kind = CATEGORICAL if self.classifies else NUMERIC
        example_kinds = (NUMERIC,) + (lag_kind,) * (self.lags + 1) + tuple(
            kinds[i] if i < len(kinds) else NUMERIC for i in self._known())
        self.processing_ = Processing(self.plan or {}, example_kinds)
        examples = self.processing_.fit_transform(np.array(rows))
        out = self.processing_.kinds_out()
        assert self.model is not None
        # The examples are processed already; a tuned model's fit would repeat
        # processing.
        self.fitted_ = OutputBaseModel.fit(
            self.model, self.hyperparameters, examples, np.array(targets), self.seed,
            kinds=out if out is not None else (NUMERIC,) * examples.shape[1])  # pyright: ignore[reportOptionalSubscript]
        self.classes_ = getattr(self.fitted_, "classes_", None)
        return self

    def _asked(self, X):
        X = np.asarray(X, dtype=float)
        keys, ahead = steps_ahead(X, self.forecast)
        rows, last = [], []
        for j, key in enumerate(keys):
            coded = self.history_.get(key, np.array([np.nan]))
            rows.append(self._features(coded, ahead[j], X[j]))
            last.append(coded[-1])
        return self.processing_.transform(np.array(rows)), np.asarray(last, dtype=float)

    def predict(self, X):
        examples, last = self._asked(X)
        assert self.model is not None
        predicted = self.model.predict(self.fitted_, examples)
        if self.classifies:
            return np.ravel(predicted)
        return last + np.asarray(predicted, dtype=float)

    def predict_proba(self, X):
        examples, _ = self._asked(X)
        assert self.model is not None
        return self.model.predict_proba(self.fitted_, examples)
