"""`core.metrics` scores every trial in the application, since a model scoring
itself under its own scikit-learn version could store incomparable numbers in
the `.ihpo`. Each `Metric` declares its direction and bounds, because an RMSE, a
log-loss or an imported SMAC cost breaks the old bigger-is-better-on-[0, 1]
assumption.
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

#: `LABELS` is one predicted label per row; `PROBABILITIES` is a
#: class-probability matrix plus class order, from `fit_predict_proba`. A string
#: leaves room for a third input kind without a second boolean.
LABELS = "labels"
PROBABILITIES = "probabilities"
#: MASE scales by each series' training values and season length.
HISTORY = "history"


def _zero(y_true) -> float:
    return 0.0


@dataclass(frozen=True)
class Metric:
    """`name` repeats the `METRICS` key, so a metric passed alone can name
    itself and an imported run can declare an unknown metric.
    """

    name: str
    #: Called with `(y_true, y_pred)` or `(y_true, y_proba, classes)`.
    fn: Callable[..., float]
    #: Every comparison in the application consults `higher_is_better`.
    higher_is_better: bool = True
    #: RMSE is `(0.0, None)`, safe for a log axis; a raw cost is `(None, None)`,
    #: with no absolute scale.
    bounds: tuple[Optional[float], Optional[float]] = (0.0, 1.0)
    needs: str = LABELS
    #: `null_score` takes `y_true` because a constant null score for an
    #: unbounded metric would be infinity, which JSON, `np.mean`, Plotly and a
    #: surrogate all reject.
    null_score: Callable[[Any], float] = field(default=_zero)
    #: A metric of the wrong task is meaningless, like the accuracy of a
    #: regression.
    tasks: tuple[str, ...] = (CLASSIFICATION,)
    #: A forecast metric needs a series' history or the values' scale, which
    #: unrelated rows lack.
    forecast_only: bool = False
    #: Only figures read `label`, and an imported metric shows its plain name.
    label: str = ""


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
        """`worst_bound` is the range's worst end, or None."""
        return self.low if self.higher_is_better else self.high

    @property
    def best_bound(self) -> Optional[float]:
        return self.high if self.higher_is_better else self.low

    @property
    def worst_sentinel(self) -> float:
        """Every real score beats `worst_sentinel`.
        """
        return float("-inf") if self.higher_is_better else float("inf")

    def better(self, score: float, than: float) -> bool:
        """`better` means strict improvement."""
        return score > than if self.higher_is_better else score < than

    def best(self, items: Iterable, *, key=None, default=None):
        """`best` takes `key` and `default` to replace `max`/`min` at both kinds
        of call site.
        """
        picker = max if self.higher_is_better else min
        if key is None:
            return picker(items, default=default)
        return picker(items, key=key, default=default)


def _roc_auc(y_true, proba, classes) -> float:
    """`_roc_auc` averages one-vs-rest AUC over the classes a fold can rank, and
    two classes give the binary AUC."""
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    scores = [roc_auc_score(y == label, proba[:, column])
              for column, label in enumerate(classes)
              if 0 < np.sum(y == label) < len(y)]
    if len(classes) == 2 and scores:
        return float(scores[1] if len(scores) > 1 else scores[0])
    return float(np.mean(scores)) if scores else 0.5


def _smape(y_true, y_pred) -> float:
    """0 is perfect and 200 is maximal; two zeros count as no error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y) + np.abs(p)
    terms = np.divide(2 * np.abs(y - p), denominator,
                      out=np.zeros_like(y), where=denominator > 0)
    return float(100 * np.mean(terms))


def _mase(y_true, y_pred, history) -> float:
    """`_mase` divides the forecast's MAE by the seasonal naive MAE on each
    series' history. *history* contains ``{"y", "series", "season"}``. A history
    shorter than a season scales by the one-step naive error, and a flat one
    leaves the plain MAE.
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
    """`_log_loss` gives an unseen validation label zero probability, the worst
    score for its row, and the trial stays scoreable."""
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)
    unseen = [label for label in np.unique(np.asarray(y_true)) if label not in classes]
    if unseen:
        proba = np.hstack([proba, np.zeros((len(proba), len(unseen)))])
        classes += unseen
    return float(log_loss(y_true, proba, labels=classes))


#: .ihpo files store the metric names verbatim, so renaming or redefining one
#: breaks old experiments.
METRICS: dict[str, Metric] = {
    "accuracy": Metric(
        name="accuracy",
        fn=lambda y, yp: float(accuracy_score(y, yp)),
    ),
    "f1": Metric(
        name="f1",
        fn=lambda y, yp: float(f1_score(y, yp, average="weighted", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "precision": Metric(
        name="precision",
        fn=lambda y, yp: float(precision_score(y, yp, average="weighted", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "recall(macro)": Metric(
        name="recall(macro)",
        fn=lambda y, yp: float(recall_score(y, yp, average="macro", zero_division=0)),  # pyright: ignore[reportArgumentType]
    ),
    "balanced_accuracy": Metric(
        name="balanced_accuracy",
        label="Balanced accuracy",
        fn=lambda y, yp: float(balanced_accuracy_score(y, yp)),
    ),
    # Probabilities cost an SVM several plain fits per trial, so only a model
    # that offers them and a run that asks get these.
    "roc_auc": Metric(
        name="roc_auc",
        label="ROC AUC",
        fn=_roc_auc,
        needs=PROBABILITIES,
        null_score=lambda y: 0.5,
    ),
    "log_loss": Metric(
        name="log_loss",
        label="Log loss",
        fn=_log_loss,
        higher_is_better=False,
        bounds=(0.0, None),
        needs=PROBABILITIES,
        null_score=lambda y: _prior_log_loss(y),
    ),
    # `metrics_for` lists MASE first for a forecast, and a forecast with MASE
    # under 1 beats repeating last season.
    "mase": Metric(
        name="mase",
        label="MASE",
        fn=_mase,
        higher_is_better=False,
        bounds=(0.0, None),
        needs=HISTORY,
        # A seasonal naive forecast scores about 1 by construction.
        null_score=lambda y: 1.0,
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    "smape": Metric(
        name="smape",
        label="sMAPE (%)",
        fn=_smape,
        higher_is_better=False,
        bounds=(0.0, 200.0),
        null_score=lambda y: _smape(y, np.full(len(y), np.mean(np.asarray(y, dtype=float)))),
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    # `null_score` here equals the error of predicting the mean, which is the R²
    # baseline.
    "rmse": Metric(
        name="rmse",
        label="RMSE",
        fn=lambda y, yp: float(root_mean_squared_error(y, yp)),
        higher_is_better=False,
        bounds=(0.0, None),
        null_score=lambda y: float(np.std(np.asarray(y, dtype=float))),
        tasks=(REGRESSION,),
    ),
    "mae": Metric(
        name="mae",
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
        label="R²",
        fn=lambda y, yp: float(r2_score(y, yp)),
        bounds=(None, 1.0),
        tasks=(REGRESSION,),
    ),
}


def metrics_for(task: str, *, probabilities: bool = False, forecast: bool = False) -> list[str]:
    """The first name is a new experiment's default objective. Probability
    metrics need *probabilities*, and a *forecast* puts its own metrics first.
    """
    names = [name for name, metric in METRICS.items()
             if task in metric.tasks and (forecast or not metric.forecast_only)]
    names.sort(key=lambda name: not METRICS[name].forecast_only)
    return usable(names, probabilities=probabilities)


def usable(names: Iterable[str], *, probabilities: bool) -> list[str]:
    """An unscoreable metric would fail every trial, so `usable` settles the
    list before the first.
    """
    return [name for name in names
            if probabilities or metric_for(name).needs != PROBABILITIES]


def _prior_log_loss(y_true) -> float:
    """`_prior_log_loss` scores a model only knowing the class balance."""
    _, counts = np.unique(np.asarray(y_true), return_counts=True)
    shares = counts / counts.sum()
    return float(-np.sum(shares * np.log(shares)))


def unscoreable(*_args):
    raise ValueError(
        "this metric is a reading of a stored number, not a way to compute one")


#: Older files assumed `ASSUMED`'s reading of every score. `unscoreable` lets an
#: unknown metric be shown and compared but never computed.
ASSUMED = Metric(name="", fn=unscoreable)


def metric_for(name: str) -> Metric:
    """`metric_for` spares callers a branch on unknown metrics.
    """
    return METRICS.get(name) or ASSUMED


def resolve(names: Iterable[str]) -> dict[str, Metric]:
    missing = [n for n in names if n not in METRICS]
    if missing:
        raise ValueError(f"unknown metric(s): {', '.join(missing)}")
    return {n: METRICS[n] for n in names}


# A file may declare its own metrics, so an imported run with an unknown
# objective still opens. Declarations sit beside the numbers, since a stored
# cost needs its direction.


def describe(metric: Metric) -> dict:
    """`describe` keeps the name, direction and range, and drops `fn` and
    `null_score`.
    """
    return {"name": metric.name,
            "higher_is_better": metric.higher_is_better,
            "bounds": list(metric.bounds),
            "needs": metric.needs}


def undescribe(raw: Mapping[str, Any]) -> Metric:
    """`undescribe` builds an unscoreable metric."""
    low, high = (list(raw.get("bounds") or [None, None]) + [None, None])[:2]
    return Metric(
        name=str(raw.get("name") or ""),
        fn=unscoreable,
        higher_is_better=bool(raw.get("higher_is_better", True)),
        bounds=(low, high),
        needs=str(raw.get("needs") or LABELS),
    )


def declarations(raw) -> dict[str, Metric]:
    """A malformed entry falls back to `ASSUMED` for its own metric alone.
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
    """*y_pred* may be a list, as from a model in another process, but the label
    type must match the target. A metric missing its probabilities or *history*
    raises.
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


# SMAC wants a cost to minimise, and `to_cost` derives one from the metric
# itself. The four original metrics come out at `1.0 - score`, and a metric that
# is already a cost round-trips unchanged.


def to_error(metric: Metric, score: float) -> Optional[float]:
    """An unbounded higher-is-better metric has no fixed best, so `to_error`
    returns None and the figure shows its empty caption. `to_error` passes a
    lower-is-better score through.
    """
    if not metric.higher_is_better:
        return score
    if metric.high is None:
        return None
    return metric.high - score


def error_range(metric: Metric) -> Optional[tuple[float, float]]:
    """None hides the absolute-scale toggle for an unbounded metric.
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
        # With no upper bound, negation reverses the order without loss.
        return -score
    return metric.high - score


def from_cost(metric: Metric, cost: float) -> float:
    if not metric.higher_is_better:
        return cost
    if metric.high is None:
        return -cost
    return metric.high - cost


def scale(metric: Metric, null_score: Optional[float] = None) -> Optional[float]:
    """`scale` returns the range of a bounded metric, otherwise the distance
    from the best end to *null_score*, or None.
    """
    if metric.low is not None and metric.high is not None:
        return metric.high - metric.low
    if metric.best_bound is None or null_score is None:
        return None
    distance = abs(float(null_score) - metric.best_bound)
    return distance if distance > 0 and math.isfinite(distance) else None
