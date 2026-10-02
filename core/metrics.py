"""The metrics every trial is scored with, and what each one is.

Scoring is the application's job, not the model's. A model returns predictions
and never sees the validation labels, so a built-in model, an uploaded one, and
one running in its own environment are all measured by exactly this code —
which is what makes scores comparable between them. Were each model to score
itself, two models resolving different scikit-learn versions could return
different numbers for identical predictions, and that difference would be
stored in the `.ihpo` with no sign of where it came from.

They live in `core` rather than `ui` because the optimizers do the scoring, and
`core` may not import the web layer.

A metric used to be a bare callable, and everything downstream assumed what the
four of them happened to share: a number on [0, 1] where bigger is better. That
assumption is spread across the incumbent, the stopping criteria, the figures'
axes, the colour scales, and the cost SMAC is asked to minimise — and it is
wrong for an RMSE (lower is better, unbounded above), for a log-loss, and for a
raw SMAC cost imported from somebody else's run. So a metric now says what it
is, and the rest of the application asks rather than assumes.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)

from .tasks import CLASSIFICATION, REGRESSION

#: What a metric needs handed to it. `LABELS` is one predicted label per row —
#: what `fit_predict` has always returned. `PROBABILITIES` is a per-class
#: probability matrix plus the class order, which a metric like ROC AUC needs
#: and which only a model that offers `fit_predict_proba` can supply.
#:
#: A string rather than a `needs_proba` boolean: a third kind of input (decision
#: scores, per-row weights) should not mean a second boolean and a rule about
#: which wins.
LABELS = "labels"
PROBABILITIES = "probabilities"
#: A forecast's predictions plus what was known before them: each series'
#: training values and the season length, which MASE scales its error by.
HISTORY = "history"


def _zero(y_true) -> float:
    """What the four original metrics score a trial that produced nothing.

    Named rather than inlined so `Metric.null_score`'s default is a plain
    function; see that field for why it is a callable at all.
    """
    return 0.0


@dataclass(frozen=True)
class Metric:
    """One way of scoring a trial, and everything the application needs to know
    about the number it produces.

    `name` repeats the key it is stored under in `METRICS`. Deliberate: a metric
    is passed around on its own — into `evaluate_trial`, into a plot builder,
    into an error message — and it has to be able to say what it is without its
    dictionary. It is also what makes a metric *serializable*, which is how an
    imported run can declare a metric this build has never heard of.
    """

    name: str
    #: `(y_true, y_pred) -> float` for a LABELS metric; `(y_true, y_proba,
    #: classes) -> float` for a PROBABILITIES one.
    fn: Callable[..., float]
    #: Which direction is an improvement. One boolean, consulted by every
    #: comparison in the application — the incumbent, the stopping criteria, the
    #: delta arrow, the colour scale. Not an enum: there are two directions and
    #: there always will be.
    higher_is_better: bool = True
    #: `(low, high)`, either end `None` for unbounded. A pair rather than an
    #: `is_bounded` flag because the two ends are asked about separately: RMSE is
    #: `(0.0, None)` and a log axis over it is safe, while a raw cost is
    #: `(None, None)` and the absolute-scale toggle has nothing to offer.
    bounds: tuple[Optional[float], Optional[float]] = (0.0, 1.0)
    #: LABELS or PROBABILITIES.
    needs: str = LABELS
    #: What a trial that produced no measurement scores, given the true labels.
    #:
    #: A callable of `y_true`, not a constant, because the honest constant for an
    #: unbounded metric is infinity — which JSON cannot carry, `np.mean` cannot
    #: average, Plotly cannot draw and a surrogate cannot fit. "As badly as a
    #: model that knows nothing" is deterministic, derived from the data rather
    #: than invented, and for all four original metrics it is exactly the 0.0
    #: they have always scored.
    null_score: Callable[[Any], float] = field(default=_zero)
    #: Which tasks this metric scores (`core.tasks`). A metric of the wrong
    #: task is not merely uninformative but meaningless — the accuracy of a
    #: regression is the share of exactly-right real numbers.
    tasks: tuple[str, ...] = (CLASSIFICATION,)
    #: Only meaningful for a forecast (`core.forecasting`): scored against the
    #: series' own history, or as an error relative to the values' size, which
    #: a table of unrelated rows has no sense of.
    forecast_only: bool = False
    #: What a figure calls it, for a name that capitalizing does not make
    #: readable — an acronym, a superscript. Presentation only: a file does not
    #: carry it, so a metric declared by an imported run reads as its name.
    label: str = ""

    # ── what the rest of the application asks ────────────────────────────────

    @property
    def title(self) -> str:
        """The metric as an axis or a colour bar names it."""
        return self.label or self.name.capitalize()

    @property
    def low(self) -> Optional[float]:
        return self.bounds[0]

    @property
    def high(self) -> Optional[float]:
        return self.bounds[1]

    @property
    def worst_bound(self) -> Optional[float]:
        """The end of the range a perfect failure would sit at, or None."""
        return self.low if self.higher_is_better else self.high

    @property
    def best_bound(self) -> Optional[float]:
        return self.high if self.higher_is_better else self.low

    @property
    def worst_sentinel(self) -> float:
        """The neutral element for "best so far": every real score beats it.

        What `float("-inf")` was, when everything was a maximisation.
        """
        return float("-inf") if self.higher_is_better else float("inf")

    def better(self, score: float, than: float) -> bool:
        """Is *score* an improvement on *than*? Strictly."""
        return score > than if self.higher_is_better else score < than

    def best(self, items: Iterable, *, key=None, default=None):
        """The best of *items* — `max` or `min`, whichever this metric means.

        Takes `key` and `default` because it stands in for `max`/`min` at call
        sites that used both: the best *score* out of a generator, and the best
        *trial* out of a list.
        """
        picker = max if self.higher_is_better else min
        if key is None:
            return picker(items, default=default)
        return picker(items, key=key, default=default)


#: Metric name → `Metric`. The names are stored verbatim in .ihpo files, so
#: renaming one breaks every experiment that used it — and changing what one
#: *means* is worse, since nothing in the file would say so. The four here are
#: the ones that predate metrics describing themselves, and they are declared
#: with exactly the assumptions the whole application used to make about them.
METRICS: dict[str, Metric] = {
    "accuracy": Metric(
        name="accuracy",
        fn=lambda y, yp: float(accuracy_score(y, yp)),
    ),
    "f1": Metric(
        name="f1",
        fn=lambda y, yp: float(f1_score(y, yp, average="weighted", zero_division=0)),
    ),
    "precision": Metric(
        name="precision",
        fn=lambda y, yp: float(precision_score(y, yp, average="weighted", zero_division=0)),
    ),
    "recall(macro)": Metric(
        name="recall(macro)",
        fn=lambda y, yp: float(recall_score(y, yp, average="macro", zero_division=0)),
    ),
    "balanced_accuracy": Metric(
        name="balanced_accuracy",
        label="Balanced accuracy",
        fn=lambda y, yp: float(balanced_accuracy_score(y, yp)),
    ),
    # Scored on probabilities, so only for a model that offers them, and only
    # when asked for: an SVM pays for its probabilities with an internal
    # calibration costing several plain fits, on every trial.
    "roc_auc": Metric(
        name="roc_auc",
        label="ROC AUC",
        fn=lambda y, proba, classes: _roc_auc(y, proba, classes),
        needs=PROBABILITIES,
        null_score=lambda y: 0.5,
    ),
    "log_loss": Metric(
        name="log_loss",
        label="Log loss",
        fn=lambda y, proba, classes: _log_loss(y, proba, classes),
        higher_is_better=False,
        bounds=(0.0, None),
        needs=PROBABILITIES,
        null_score=lambda y: _prior_log_loss(y),
    ),
    # Forecasting. MASE first, so it is what a forecast is optimized on unless
    # asked otherwise: it is the one that means the same on any series —
    # below 1 beats repeating last season, above 1 does not.
    "mase": Metric(
        name="mase",
        label="MASE",
        fn=lambda y, yp, history: _mase(y, yp, history),
        higher_is_better=False,
        bounds=(0.0, None),
        needs=HISTORY,
        # The no-skill line by construction: a seasonal naive forecast scores
        # about 1 on the history it is scaled by.
        null_score=lambda y: 1.0,
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    "smape": Metric(
        name="smape",
        label="sMAPE (%)",
        fn=lambda y, yp: _smape(y, yp),
        higher_is_better=False,
        bounds=(0.0, 200.0),
        null_score=lambda y: _smape(y, np.full(len(y), np.mean(np.asarray(y, dtype=float)))),
        tasks=(REGRESSION,),
        forecast_only=True,
    ),
    # Regression. A trial that produced nothing scores what predicting the
    # mean scores: the error of a model that knows only the target's average,
    # which is what R² measures every model against anyway.
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
    """The names of the metrics that score *task*, in the order they are listed.

    The first is what a new experiment optimizes unless told otherwise. The
    ones scored on probabilities are left out unless *probabilities*: they are
    asked for, not given — see `roc_auc`. A *forecast* also gets the ones only a
    forecast has, first, so it is optimized on MASE unless told otherwise.
    """
    names = [name for name, metric in METRICS.items()
             if task in metric.tasks and (forecast or not metric.forecast_only)]
    names.sort(key=lambda name: not METRICS[name].forecast_only)
    return usable(names, probabilities=probabilities)


def usable(names: Iterable[str], *, probabilities: bool) -> list[str]:
    """*names* without the metrics that need probabilities, unless the model
    scoring them can give *probabilities*.

    Settled before the first trial, because a metric that cannot be scored
    does not fail one trial — it fails every one of them.
    """
    return [name for name in names
            if probabilities or metric_for(name).needs != PROBABILITIES]


def _roc_auc(y_true, proba, classes) -> float:
    """ROC AUC, one class against the rest, averaged over the classes this
    fold can rank — a class with no rows in the fold, or every row, has no
    curve. Two classes is the usual binary AUC."""
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    scores = [roc_auc_score(y == label, proba[:, column])
              for column, label in enumerate(classes)
              if 0 < np.sum(y == label) < len(y)]
    if len(classes) == 2 and scores:
        return float(scores[1] if len(scores) > 1 else scores[0])
    return float(np.mean(scores)) if scores else 0.5


def _smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error, in percent: 0 is perfect, 200
    is as wrong as two numbers can be. A row where both are zero is no error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y) + np.abs(p)
    terms = np.divide(2 * np.abs(y - p), denominator,
                      out=np.zeros_like(y), where=denominator > 0)
    return float(100 * np.mean(terms))


def _mase(y_true, y_pred, history) -> float:
    """Mean absolute scaled error: the forecast's mean absolute error over that
    of a seasonal naive forecast — this season's value next season — on each
    series' own history, averaged over the series.

    *history* is ``{"y", "series", "season"}``: the training values in time
    order, each one's series (None for one series), and the season length. A
    history too short for a whole season is scaled by the one-step naive
    forecast instead, and a flat one, whose naive forecast is never wrong, by
    nothing — its MASE is its plain mean absolute error.
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
    """Log loss, counting a validation label the model never saw in training as
    one it gave no probability to — the worst it can score for that row, rather
    than a trial that cannot be scored at all. It happens whenever a class is
    rarer than the folds, or only appears late in a series divided in time."""
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)
    unseen = [label for label in np.unique(np.asarray(y_true)) if label not in classes]
    if unseen:
        proba = np.hstack([proba, np.zeros((len(proba), len(unseen)))])
        classes += unseen
    return float(log_loss(y_true, proba, labels=classes))


def _prior_log_loss(y_true) -> float:
    """The log loss of predicting each class's share of *y_true* for every
    row: what a model that knows only the class balance scores."""
    _, counts = np.unique(np.asarray(y_true), return_counts=True)
    shares = counts / counts.sum()
    return float(-np.sum(shares * np.log(shares)))


def _unscoreable(*_args):
    raise ValueError(
        "this metric is a reading of a stored number, not a way to compute one")


#: How a name nobody recognises is read. Every version of this code before
#: metrics could describe themselves assumed exactly this of every score, so a
#: file written then is read back exactly as it was written. Its `fn` refuses:
#: an unknown metric can be *displayed* and *compared*, but a run cannot be
#: scored by it, and the difference should not be discoverable by surprise.
ASSUMED = Metric(name="", fn=_unscoreable)


def metric_for(name: str) -> Metric:
    """*name*'s metric, or `ASSUMED` — never None.

    For the many call sites that need to know a direction and have no business
    branching on whether the metric is one this build ships.
    """
    return METRICS.get(name) or ASSUMED


def lookup(name: str) -> Optional[Metric]:
    """*name*'s metric, or None when this build has never heard of it.

    None is a real answer rather than an error: a file can name a metric that
    only exists inside it (an imported run's own objective), and the caller
    decides whether that is fatal or something to fall back from.
    """
    return METRICS.get(name)


def resolve(names: Iterable[str]) -> dict[str, Metric]:
    """``{name: Metric}`` for *names*, or ValueError naming the ones that don't exist."""
    missing = [n for n in names if n not in METRICS]
    if missing:
        raise ValueError(f"unknown metric(s): {', '.join(missing)}")
    return {n: METRICS[n] for n in names}


# ── metrics a file brings with it ────────────────────────────────────────────
#
# A run imported from somewhere else names an objective this build has never
# heard of, and refusing it would mean the file cannot be opened at all. So a
# file may *declare* its own metrics: enough to read, compare and draw the
# numbers it stores, which is everything except computing new ones.
#
# Declared beside the numbers rather than in a section of its own, so the two
# cannot be separated — a stored cost is meaningless without the direction it
# was stored under.


def describe(metric: Metric) -> dict:
    """*metric* as JSON, for a file to carry.

    `fn` and `null_score` are code and do not travel. What survives is what a
    reader needs: the name, the direction and the range.
    """
    return {"name": metric.name,
            "higher_is_better": metric.higher_is_better,
            "bounds": list(metric.bounds),
            "needs": metric.needs}


def undescribe(raw: Mapping[str, Any]) -> Metric:
    """A declaration read back. Unscoreable by construction — see `ASSUMED`."""
    low, high = (list(raw.get("bounds") or [None, None]) + [None, None])[:2]
    return Metric(
        name=str(raw.get("name") or ""),
        fn=_unscoreable,
        higher_is_better=bool(raw.get("higher_is_better", True)),
        bounds=(low, high),
        needs=str(raw.get("needs") or LABELS),
    )


def declarations(raw) -> dict[str, Metric]:
    """`{name: Metric}` from a stored `declared_metrics` block, or `{}`.

    Tolerant on purpose: this is read out of a file, and a malformed entry
    should cost that one metric its declaration — which falls back to `ASSUMED`,
    the reading every older file was written under — rather than the whole file.
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
    """Score one set of predictions against every metric in *metrics*.

    *y_pred* may be a plain list rather than an array — that is how predictions
    come back from a model in another process. The scorers accept either, as
    long as the label *type* matches: a model returning strings for an integer
    target fails here, which is the right place for it to fail.

    *y_proba* and *classes* are the probability matrix and its column order,
    supplied only when the model can produce them. A metric that needs them and
    is asked to score without them raises. *history* is what a forecast was
    made from (see `_mase`), for the metrics that need it.
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


# ── the score an optimizer minimises ─────────────────────────────────────────
#
# SMAC minimises. The application maximises, or did — so somewhere the two have
# to meet, and for years that was a hardcoded `1.0 - score` at six call sites.
#
# It is derived here instead, from what the metric says about itself, so that
# the four original metrics come out at exactly `1.0 - score` and every `.ihpo`
# ever written keeps the cost it has. No convention flag in the file, no format
# bump, and a metric that is already a cost — an imported SMAC objective — round
# trips through both functions unchanged.


def to_error(metric: Optional[Metric], score: float) -> Optional[float]:
    """*score* as a distance from perfect, or None when it has no such reading.

    Not the same question as `to_cost`, though it agrees with it on two of the
    three cases. A cost only has to *reverse* the ordering, so negating an
    unbounded score is a perfectly good cost; an error has to be a distance from
    a fixed best, and an unbounded metric has no best to measure from. There is
    no honest number to return there, so there is no number — the figure shows
    its empty caption rather than an axis measured from nowhere.

    A metric that is already a cost (lower wins) is already an error, and
    passes through.
    """
    if metric is None:
        # The one convention there was before metrics described themselves.
        return 1.0 - score
    if not metric.higher_is_better:
        return score
    if metric.high is None:
        return None
    return metric.high - score


def error_range(metric: Optional[Metric]) -> Optional[tuple[float, float]]:
    """The span an error axis covers for *metric*, or None if it has no fixed one.

    What the absolute-scale toggle pins to. None means the toggle has nothing to
    offer and hides — which is the honest answer for an unbounded metric, where
    the only range there is is the one the data happens to occupy.
    """
    if metric is None:
        return (0.0, 1.0)
    if metric.low is None or metric.high is None:
        return None
    if metric.higher_is_better:
        return (0.0, metric.high - metric.low)
    return (metric.low, metric.high)


def to_cost(metric: Optional[Metric], score: float) -> float:
    """*score* as something to minimise."""
    if metric is None:
        # A name this build has never heard of, in a file old enough to predate
        # per-metric scores. Every such file was written under the one
        # convention there was.
        return 1.0 - score
    if not metric.higher_is_better:
        return score
    if metric.high is None:
        # Nothing to subtract from. Negating is the only order-reversing map
        # left, and it is exact.
        return -score
    return metric.high - score


def from_cost(metric: Optional[Metric], cost: float) -> float:
    """The inverse of `to_cost`, exactly."""
    if metric is None:
        return 1.0 - cost
    if not metric.higher_is_better:
        return cost
    if metric.high is None:
        return -cost
    return metric.high - cost
