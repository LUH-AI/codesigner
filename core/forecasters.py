"""The forecasting estimators: fitted on a series' past, asked about its future.

Each is a scikit-learn-style estimator, like every other model's: `fit(X, y)`
learns from the training rows — in time order, a series' history — and
`predict(X)` forecasts the rows asked about, from what each row says of itself
(its time and its series) and what `fit` kept of its series' past. What a
forecaster is told about the data is `forecast` (see `core.forecasting.context`):

    {"horizon", "season", "gap", "time": column of X with each row's time,
     "series": column naming its series (None for one series),
     "known": the columns of X known ahead of time}

A row's step ahead is its time's place among its series' rows being forecast,
counted from one past the gap — so the rows asked about are the next steps
after the history, as in a backtest and as in use.

Imports nothing but the delivered-model contract and the processing, so an
exported forecaster carries this module as it is.
"""

import warnings

import numpy as np

from codesigner_model import OutputBaseModel

from .processing import Processing

NUMERIC = "numeric"
CATEGORICAL = "categorical"


def series_keys(X, series):
    """Each row's series, as text, so a missing one is a series of its own."""
    if series is None:
        return np.zeros(len(X), dtype=object).astype(str)
    return np.array([str(v) for v in np.asarray(X)[:, series]])


def steps_ahead(X, forecast):
    """Each row's series and its step ahead of its series' history — 1 for the
    first time after the end of it, plus the gap."""
    X = np.asarray(X, dtype=float)
    keys = series_keys(X, forecast.get("series"))
    ahead = np.zeros(len(X), dtype=int)
    time, gap = forecast.get("time", 0), int(forecast.get("gap", 0))
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        order = {t: i for i, t in enumerate(np.unique(X[rows, time]))}
        ahead[rows] = [order[t] + 1 + gap for t in X[rows, time]]
    return keys, ahead


class _Estimator:
    """What every forecaster here shares: its parameters are its constructor's
    arguments, as scikit-learn expects."""

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._params}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _history(self, X, y):
        """Each series' values, oldest first, as the training rows give them."""
        keys = series_keys(X, self.forecast.get("series"))
        y = np.asarray(y, dtype=float)
        return {key: y[keys == key] for key in np.unique(keys)}


class SeasonalNaiveForecaster(_Estimator):
    """The forecasts to beat: last season again (``"seasonal"``), the last value
    again (``"last"``), or the straight line from the first value to the last
    (``"drift"``). Its only state is the history itself."""

    _params = ("strategy", "forecast")

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
    """Exponential smoothing (statsmodels' Holt–Winters): a level, optionally a
    trend (``"add"``, possibly damped) and a season (``"add"`` or ``"mul"``),
    each updated as the series goes. One smoother per series, fitted on its
    own history.

    A season needs two of them to be estimated, and a multiplicative one needs
    every value positive; a series without either gets the next simpler model.
    """

    _params = ("trend", "damped", "seasonal", "forecast")

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
                # Convergence and frequency notices, which nobody can act on here.
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
    """Any model, forecasting by reduction.

    Each training row becomes one example per step ahead: the step, the series'
    last *lags* values before the forecast's origin (latest first), a summary
    of the last season (its mean; for labels, its commonest), and the row's
    columns known ahead of time — never its others, whose future values nobody
    has yet. One model learns every step. For numbers it predicts the change
    from the origin's value, so a tree can follow a trend it was never shown
    the far end of; for labels, the label, from the lagged labels' codes.

    *model* is the model doing the learning (a `codesigner_model.OutputBaseModel`),
    fitted with *hyperparameters*; *plan* is the experiment's processing of the
    examples it learns from (`core.processing`), applied here because the
    examples are made here.
    """

    _params = ("model", "hyperparameters", "lags", "forecast", "plan", "task", "seed")

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
        """One example: the step, the lags, the season's summary, the row's
        known columns."""
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
            # Lagged labels as codes: a label's place among the training labels.
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
        # The delivered model's own fit: the examples are processed already,
        # and a tuned model's fit would process them again.
        self.fitted_ = OutputBaseModel.fit(
            self.model, self.hyperparameters, examples, np.array(targets), self.seed,
            kinds=out if out is not None else (NUMERIC,) * examples.shape[1])
        self.classes_ = getattr(self.fitted_, "classes_", None)
        return self

    def _asked(self, X):
        """The examples for the rows asked about, and each row's series' value
        at the origin."""
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
        predicted = self.model.predict(self.fitted_, examples)
        if self.classifies:
            return np.ravel(predicted)
        return last + np.asarray(predicted, dtype=float)

    def predict_proba(self, X):
        examples, _ = self._asked(X)
        return self.model.predict_proba(self.fitted_, examples)
