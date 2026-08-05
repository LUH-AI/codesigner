"""The metrics every trial is scored with.

Scoring is the application's job, not the model's. A model returns predictions
and never sees the validation labels, so a built-in model, an uploaded one, and
one running in its own environment are all measured by exactly this code —
which is what makes scores comparable between them. Were each model to score
itself, two models resolving different scikit-learn versions could return
different numbers for identical predictions, and that difference would be
stored in the `.ihpo` with no sign of where it came from.

They live in `core` rather than `ui` because the optimizers do the scoring, and
`core` may not import the web layer.
"""

from collections.abc import Callable, Iterable, Mapping

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#: Metric name → scorer, `(y_true, y_pred) -> float`. The names are stored
#: verbatim in .ihpo files, so renaming one breaks every experiment that used it.
METRICS: dict[str, Callable[..., float]] = {
    "accuracy":      lambda y, yp: float(accuracy_score(y, yp)),
    "f1":            lambda y, yp: float(f1_score(y, yp, average="weighted", zero_division=0)),
    "precision":     lambda y, yp: float(precision_score(y, yp, average="weighted", zero_division=0)),
    "recall(macro)": lambda y, yp: float(recall_score(y, yp, average="macro", zero_division=0)),
}


def resolve(names: Iterable[str]) -> dict[str, Callable[..., float]]:
    """``{name: scorer}`` for *names*, or ValueError naming the ones that don't exist."""
    missing = [n for n in names if n not in METRICS]
    if missing:
        raise ValueError(f"unknown metric(s): {', '.join(missing)}")
    return {n: METRICS[n] for n in names}


def score_all(y_true, y_pred, metrics: Mapping[str, Callable[..., float]]) -> dict[str, float]:
    """Score one set of predictions against every metric in *metrics*.

    *y_pred* may be a plain list rather than an array — that is how predictions
    come back from a model in another process. The scorers accept either, as
    long as the label *type* matches: a model returning strings for an integer
    target fails here, which is the right place for it to fail.
    """
    return {name: float(fn(y_true, y_pred)) for name, fn in metrics.items()}
