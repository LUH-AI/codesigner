"""How a trial's data is divided, whether that is one holdout or k folds.

A single holdout is fast and is all a large dataset needs. On the small tables
people actually upload it is a noisy estimate, and a search that optimizes a
noisy estimate spends its budget chasing the split rather than the model.
Cross-validation trades k times the compute for an average that means something.

Both are expressed the same way — **a list of (train, val) index pairs** —
because everything downstream then has one shape to handle: the trial loop
iterates folds, the wire protocol sends folds, and holdout is simply the case
where there is one. The alternative, a boolean and two code paths, would have
had to be threaded through the optimizers, the subprocess protocol and the
harness twice over.

It also keeps a property worth keeping. A model in its own process is sent the
training labels for each fold and never the labels it is about to be scored on.
With one fold that is the whole guarantee: the model cannot see the answers.
With k folds every row is a validation row exactly once, so across the folds of
a single trial their union is all of them, and a model that deliberately cached
what it was sent could reconstruct the labels. It cannot do so *for the fold
being scored*, which is what stops an accidental leak, but this is a weaker
statement and should not be read as the stronger one. Closing it would mean one
process per fold — k times the memory for the whole run — and that is not a
trade worth making by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .tasks import CLASSIFICATION

#: Below this many folds it is not cross-validation. Stored as 0 on an
#: experiment that uses a single holdout.
MIN_FOLDS = 2


@dataclass(frozen=True)
class Splits:
    """The whole dataset, and the folds a trial is evaluated over.

    A forecast's backtests (`backtest`) also say which series each row is of,
    the season length and the horizon: MASE scales its error by a seasonal
    naive forecast of each series' own history.
    """

    X: Any
    y: Any
    folds: List[Tuple[np.ndarray, np.ndarray]]
    series: Any = None
    season: int = 1
    horizon: int = 0

    @property
    def is_cv(self) -> bool:
        return len(self.folds) > 1


def holdout(X_train, y_train, X_val, y_val) -> Splits:
    """Today's single 80/20 split, expressed as one fold.

    The arrays are already divided, so they are concatenated back and indexed —
    train first, validation second — rather than re-splitting the frame. That
    keeps this exactly the split `_load_splits` produced, including its
    stratification, instead of a second attempt at reproducing it.
    """
    n_train, n_val = len(X_train), len(X_val)
    X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
    # No dtype forced. An integer target coerced to object stops looking like
    # discrete classes to scikit-learn — `type_of_target` calls it "unknown" and
    # every classifier refuses to fit, so every trial fails at once.
    y = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    return Splits(X=X, y=y, folds=[(train_idx, val_idx)])


def cross_validation(X, y, folds: int, seed: int, task: str = CLASSIFICATION) -> Splits:
    """*folds*-fold cross-validation over the whole dataset.

    For classification, stratified when the labels allow it and a plain
    shuffle when they do not. A class with fewer members than there are folds
    only earns a warning and is still stratified. For regression, always a
    plain shuffle. Seeded, so the same experiment divides the same way every
    run.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)          # dtype preserved — see `holdout` for why

    divided = None
    if task == CLASSIFICATION:
        try:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            divided = list(splitter.split(X, y))
        except ValueError:
            pass
    if divided is None:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        divided = list(splitter.split(X))

    return Splits(X=X, y=y, folds=divided)


# ── ordered by time ───────────────────────────────────────────────────────────
#
# When the rows are moments in time, a random division lets a model train on
# next year to predict last year, and scores it on an easier problem than the
# one it will be used for. Divided in time order instead, every validation row
# comes after every row the model learnt from. Never shuffled and never
# stratified: either would put the future back into the training rows.
#
# Both take rows already sorted by time (`core.io` sorts them), so a fold is a
# range of positions.


def time_holdout(X, y, test_size: float, gap: int = 0) -> Splits:
    """The latest *test_size* of the rows validated, the rest trained on.

    *gap* rows between the two are used for neither — what a deployed model
    would not yet have, when its labels arrive late.
    """
    X, y = np.asarray(X), np.asarray(y)
    n = len(X)
    n_val = max(1, int(np.ceil(n * float(test_size))))
    n_train = n - n_val - int(gap)
    if n_train < 1:
        raise ValueError(f"{n} rows leave nothing to train on before the latest "
                         f"{n_val} with a gap of {gap}")
    return Splits(X=X, y=y, folds=[(np.arange(n_train), np.arange(n - n_val, n))])


def time_cross_validation(X, y, folds: int, gap: int = 0) -> Splits:
    """Rolling-origin cross-validation: *folds* successive blocks of the latest
    rows, each validated after training on everything before it (less *gap*).

    The first block of rows is only ever trained on, so unlike shuffled
    cross-validation not every row is validated once — there is nothing before
    the earliest rows to have learnt them from.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X, y = np.asarray(X), np.asarray(y)
    try:
        divided = list(TimeSeriesSplit(n_splits=folds, gap=int(gap)).split(X))
    except ValueError as exc:
        raise ValueError(f"{len(X)} rows cannot be divided into {folds} time-ordered "
                         f"folds with a gap of {gap}: {exc}") from exc
    return Splits(X=X, y=y, folds=divided)


# ── forecasting: backtests from rolling origins ──────────────────────────────


def backtest(X, y, times, horizon: int, folds: int, gap: int = 0,
             series=None, season: int = 1) -> Splits:
    """*folds* forecasts of the next *horizon* time steps, each from an origin
    one horizon later than the last, ending at the latest time in the data.

    Over the distinct *times* rather than the rows, so several series sharing
    their timestamps are cut at the same moment and each forecasts *horizon*
    steps of its own. A fold trains on every row before its origin, less *gap*
    time steps, and validates the next *horizon* steps. *X*, *y*, *times* and
    *series* are row-aligned and in time order.

    Only a series with history before the origin is validated: one that first
    appears inside the window has nothing to forecast from, and its rows would
    otherwise fail every trial they are in.
    """
    X, y, times = np.asarray(X), np.asarray(y), np.asarray(times)
    steps = np.unique(times)
    needed = folds * horizon + int(gap) + 1
    if len(steps) < needed:
        raise ValueError(f"{len(steps)} time steps are too few for {folds} backtests of "
                         f"{horizon} steps with a gap of {gap}: {needed} are needed")
    position = np.searchsorted(steps, times)
    divided = []
    for k in range(folds, 0, -1):
        origin = len(steps) - k * horizon
        train = np.flatnonzero(position < origin - int(gap))
        val = np.flatnonzero((position >= origin) & (position < origin + horizon))
        if series is not None:
            known = np.asarray(series)[train]
            val = val[np.isin(np.asarray(series)[val], known)]
        if not len(val):
            raise ValueError(f"the backtest from step {origin} has no series with "
                             f"history before it to forecast")
        divided.append((train, val))
    return Splits(X=X, y=y, folds=divided,
                  series=None if series is None else np.asarray(series),
                  season=max(1, int(season)), horizon=int(horizon))
