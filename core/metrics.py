"""`core.metrics` scores every trial, so numbers in one `.ihpo` stay comparable.
Each `Metric` declares its direction and bounds.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)

from .tasks import CLASSIFICATION, REGRESSION

#: `LABELS` is one label per row; `PROBABILITIES` is a class-probability matrix
#: from `.fit_predict_proba`.
LABELS = "labels"
PROBABILITIES = "probabilities"
#: MASE scales by training values and season.
HISTORY = "history"


def _zero(y_true) -> float:
    return 0.0


@dataclass(frozen=True)
class Metric:
    """`Metric` describes one score; `name` repeats its `METRICS` key.
    """

    name: str
    fn: Callable[..., float]
    higher_is_better: bool = True
    #: RMSE has `(0.0, None)`; a raw cost has `(None, None)`.
    bounds: tuple[Optional[float], Optional[float]] = (0.0, 1.0)
    needs: str = LABELS
    #: `null_score` takes `y_true`, since a constant null score for an unbounded
    #: metric would be infinity.
    null_score: Callable[[Any], float] = field(default=_zero)
    tasks: tuple[str, ...] = (CLASSIFICATION,)
    #: `forecast_only` metrics need series history, which plain rows lack.
    forecast_only: bool = False
    #: `label` titles figures, and imported metrics have none.
    label: str = ""
    #: `definition` explains the score for the assistant to quote.
    definition: str = ""


    @property
    def title(self) -> str:
        return self.label or self.name.capitalize()

    @property
    def low(self) -> Optional[float]:
        return self.bounds[0]

    @property
    def high(self) -> Optional[float]:
        return self.bounds[1]

    @property
    def worst_bound(self) -> Optional[float]:
        """`worst_bound` can be None."""
        return self.low if self.higher_is_better else self.high

    @property
    def best_bound(self) -> Optional[float]:
        return self.high if self.higher_is_better else self.low

    @property
    def worst_sentinel(self) -> float:
        return float("-inf") if self.higher_is_better else float("inf")

    def better(self, score: float, than: float) -> bool:
        """`better` means strict improvement."""
        return score > than if self.higher_is_better else score < than

    def best(self, items: Iterable, *, key=None, default=None):
        """`best` replaces `max`/`min` and takes their `key` and `default`.
        """
        picker = max if self.higher_is_better else min
        if key is None:
            return picker(items, default=default)
        return picker(items, key=key, default=default)


def _roc_auc(y_true, proba, classes) -> float:
    """`_roc_auc` averages one-vs-rest AUC over the classes a fold can rank."""
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    scores = [roc_auc_score(y == label, proba[:, column])
              for column, label in enumerate(classes)
              if 0 < np.sum(y == label) < len(y)]
    if len(classes) == 2 and scores:
        return float(scores[1] if len(scores) > 1 else scores[0])
    return float(np.mean(scores)) if scores else 0.5


def _smape(y_true, y_pred) -> float:
    """`_smape` scores two zeros as no error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y) + np.abs(p)
    terms = np.divide(2 * np.abs(y - p), denominator,
                      out=np.zeros_like(y), where=denominator > 0)
    return float(100 * np.mean(terms))


def _mase(y_true, y_pred, history) -> float:
    """`_mase` divides the forecast MAE by the series' mean seasonal naive MAE.
    A history within one season uses lag 1.
    """
    y_hist = np.asarray(history["y"], dtype=float)
    series = history.get("series")
    groups = ([y_hist] if series is None
              else [y_hist[np.asarray(series) == s] for s in np.unique(series)])
    season = max(1, int(history.get("season") or 1))
    scales = []
    for values in groups:
        lag = season if len(values) > season else 1
        if len(values) > lag:
            scales.append(np.mean(np.abs(values[lag:] - values[:-lag])))
    scale = float(np.mean(scales)) if scales else 0.0
    error = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
    return error / scale if scale > 0 else error


def _log_loss(y_true, proba, classes) -> float:
    """`_log_loss` keeps the trial scoreable by giving an unseen validation
    label zero probability."""
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)
    unseen = [label for label in np.unique(np.asarray(y_true)) if label not in classes]
    if unseen:
        proba = np.hstack([proba, np.zeros((len(proba), len(unseen)))])
        classes += unseen
    return float(log_loss(y_true, proba, labels=classes))


#: Renaming or redefining a metric breaks old `.ihpo` files.
METRICS: dict[str, Metric] = {
    "accuracy": Metric(
        name="accuracy",
        definition='The share of rows predicted correctly, so the largest classes count most.',
        fn=lambda y, yp: float(accuracy_score(y, yp)),
    ),
    "f1": Metric(
        name="f1",
        definition="F1 per class, averaged weighted by each class's size (scikit-learn's average='weighted'), so large classes dominate it.",
        fn=lambda y, yp: float(f1_score(y, yp, average="weighted", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "precision": Metric(
        name="precision",
        definition="Precision per class, averaged weighted by each class's size (average='weighted').",
        fn=lambda y, yp: float(precision_score(y, yp, average="weighted", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "recall(macro)": Metric(
        name="recall(macro)",
        definition="Recall per class, as an unweighted mean over the classes (average='macro'): each class counts equally.",
        fn=lambda y, yp: float(recall_score(y, yp, average="macro", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "balanced_accuracy": Metric(
        name="balanced_accuracy",
        definition='Recall per class, as an unweighted mean over the classes; the same number as recall(macro).',
        label="Balanced accuracy",
        fn=lambda y, yp: float(balanced_accuracy_score(y, yp)),
    ),
    # `usable` only lists probability metrics on request, since an SVM fits
    # several times for probabilities.
    "roc_auc": Metric(
        name="roc_auc",
        definition="One-vs-rest ROC AUC per class, as an unweighted mean (the binary AUC for two classes); scores the predicted probabilities' ranking.",
        label="ROC AUC",
        fn=_roc_auc,
        needs=PROBABILITIES,
        null_score=lambda y: 0.5,
    ),
    "log_loss": Metric(
        name="log_loss",
        definition='Cross-entropy of the predicted class probabilities; lower is better, and the class balance alone scores the null.',
        label="Log loss",
        fn=_log_loss,
        higher_is_better=False,
        bounds=(0.0, None),
        needs=PROBABILITIES,
        null_score=lambda y: _prior_log_loss(y),
    ),
    "mase": Metric(
        name="mase",
        definition="Mean absolute error relative to the seasonal naive forecast's; below 1 beats repeating last season.",
        label="MASE",
        fn=_mase,
        higher_is_better=False,
        bounds=(0.0, None),
        needs=HISTORY,
        # Seasonal naive scores about 1.
        null_score=lambda y: 1.0,
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    "smape": Metric(
        name="smape",
        definition='Symmetric mean absolute percentage error, from 0 (perfect) to 200.',
        label="sMAPE (%)",
        fn=_smape,
        higher_is_better=False,
        bounds=(0.0, 200.0),
        null_score=lambda y: _smape(y, np.full(len(y), np.mean(np.asarray(y, dtype=float)))),
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    # The mean predictor sets `null_score`, as in R².
    "rmse": Metric(
        name="rmse",
        definition="Root mean squared error, in the target's units; large errors weigh most.",
        label="RMSE",
        fn=lambda y, yp: float(root_mean_squared_error(y, yp)),
        higher_is_better=False,
        bounds=(0.0, None),
        null_score=lambda y: float(np.std(np.asarray(y, dtype=float))),
        tasks=(REGRESSION,),
    ),
    "mae": Metric(
        name="mae",
        definition="Mean absolute error, in the target's units.",
        label="MAE",
        fn=lambda y, yp: float(mean_absolute_error(y, yp)),
        higher_is_better=False,
        bounds=(0.0, None),
        null_score=lambda y: float(np.mean(np.abs(np.asarray(y, dtype=float)
                                                  - np.mean(np.asarray(y, dtype=float))))),
        tasks=(REGRESSION,),
    ),
    "r2": Metric(
        name="r2",
        definition="The share of the target's variance explained; 0 is predicting the mean, 1 is perfect.",
        label="R²",
        fn=lambda y, yp: float(r2_score(y, yp)),
        bounds=(None, 1.0),
        tasks=(REGRESSION,),
    ),
}


def metrics_for(task: str, *, probabilities: bool = False, forecast: bool = False) -> list[str]:
    """`metrics_for` lists a new experiment's default objective first.
    """
    names = [name for name, metric in METRICS.items()
             if task in metric.tasks and (forecast or not metric.forecast_only)]
    names.sort(key=lambda name: not METRICS[name].forecast_only)
    return usable(names, probabilities=probabilities)


def usable(names: Iterable[str], *, probabilities: bool) -> list[str]:
    """`usable` drops probability metrics, which fail every trial without
    probabilities.
    """
    return [name for name in names
            if probabilities or metric_for(name).needs != PROBABILITIES]


def _prior_log_loss(y_true) -> float:
    """`_prior_log_loss` scores the class-balance predictor."""
    _, counts = np.unique(np.asarray(y_true), return_counts=True)
    shares = counts / counts.sum()
    return float(-np.sum(shares * np.log(shares)))


def unscoreable(*_args):
    raise ValueError(
        "this metric is a reading of a stored number, not a way to compute one")


#: Older files assume `ASSUMED` for every score; its `unscoreable` `fn` raises
#: `ValueError`.
ASSUMED = Metric(name="", fn=unscoreable)


def metric_for(name: str) -> Metric:
    """`metric_for` falls back to `ASSUMED`.
    """
    return METRICS.get(name) or ASSUMED


def resolve(names: Iterable[str]) -> dict[str, Metric]:
    missing = [n for n in names if n not in METRICS]
    if missing:
        raise ValueError(f"unknown metric(s): {', '.join(missing)}")
    return {n: METRICS[n] for n in names}


# A file may declare its metrics, so an imported run with an unknown objective
# still opens.


def describe(metric: Metric) -> dict:
    """`describe` drops `fn` and `null_score`.
    """
    return {"name": metric.name,
            "higher_is_better": metric.higher_is_better,
            "bounds": list(metric.bounds),
            "needs": metric.needs}


def undescribe(raw: Mapping[str, Any]) -> Metric:
    low, high = (list(raw.get("bounds") or [None, None]) + [None, None])[:2]
    return Metric(
        name=str(raw.get("name") or ""),
        fn=unscoreable,
        higher_is_better=bool(raw.get("higher_is_better", True)),
        bounds=(low, high),
        needs=str(raw.get("needs") or LABELS),
    )


def declarations(raw) -> dict[str, Metric]:
    """`declarations` skips malformed entries.
    """
    if not isinstance(raw, Mapping):
        return {}
    out = {}
    for name, entry in raw.items():
        if isinstance(entry, Mapping):
            out[str(name)] = undescribe({"name": name, **entry})
    return out


def score_all(y_true, y_pred, metrics: Mapping[str, Metric],
              *, y_proba=None, classes=None, history=None) -> dict[str, float]:
    """*y_pred* labels must match the target's type. `score_all` raises
    `ValueError` for missing probabilities or *history*.
    """
    scores = {}
    for name, metric in metrics.items():
        if metric.needs == HISTORY:
            if history is None:
                raise ValueError(f"{name!r} needs the forecast's history and none was supplied")
            scores[name] = float(metric.fn(y_true, y_pred, history))
        elif metric.needs == PROBABILITIES:
            if y_proba is None:
                raise ValueError(
                    f"{name!r} needs class probabilities and none were supplied")
            scores[name] = float(metric.fn(y_true, y_proba, classes))
        else:
            scores[name] = float(metric.fn(y_true, y_pred))
    return scores


# `to_cost` derives the SMAC cost from the metric's direction and bounds;
# `from_cost` inverts `to_cost`.


def to_error(metric: Metric, score: float) -> Optional[float]:
    """`to_error` returns None for an unbounded higher-is-better metric and
    passes lower-is-better scores through.
    """
    if not metric.higher_is_better:
        return score
    if metric.high is None:
        return None
    return metric.high - score


def error_range(metric: Metric) -> Optional[tuple[float, float]]:
    """None hides the absolute-scale toggle.
    """
    if metric.low is None or metric.high is None:
        return None
    if metric.higher_is_better:
        return (0.0, metric.high - metric.low)
    return (metric.low, metric.high)


def to_cost(metric: Metric, score: float) -> float:
    if not metric.higher_is_better:
        return score
    if metric.high is None:
        # Negation reverses the order without loss.
        return -score
    return metric.high - score


def from_cost(metric: Metric, cost: float) -> float:
    if not metric.higher_is_better:
        return cost
    if metric.high is None:
        return -cost
    return metric.high - cost


def scale(metric: Metric, null_score: Optional[float] = None) -> Optional[float]:
    """`scale` returns the range, or the best end's distance to *null_score*.
    """
    if metric.low is not None and metric.high is not None:
        return metric.high - metric.low
    if metric.best_bound is None or null_score is None:
        return None
    distance = abs(float(null_score) - metric.best_bound)
    return distance if distance > 0 and math.isfinite(distance) else None
