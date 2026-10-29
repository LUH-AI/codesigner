"""`Splits` represents every evaluation scheme as a list of (train, val) index pairs,
so the trial loop and the wire protocol handle one shape and a holdout is the
one-fold case. Cross-validation spends k times the compute to average away the
split noise of a small table.

A model in its own process never receives the labels of the fold being scored,
yet across k folds it receives every label and could reconstruct them. One
process per fold would close the leak at k times the memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .tasks import CLASSIFICATION

#: Fewer folds means a holdout, stored as 0.
MIN_FOLDS = 2


@dataclass(frozen=True)
class Splits:
    """A forecast also carries series, season and horizon, since MASE scales
    error by each series' seasonal naive forecast.
    """

    X: Any
    y: Any
    folds: List[Tuple[np.ndarray, np.ndarray]]
    series: Any = None
    season: int = 1
    horizon: int = 0


def holdout(X_train, y_train, X_val, y_val) -> Splits:
    """`holdout` concatenates the already divided arrays, train first, to keep
    the exact stratified split `load_splits` produced.
    """
    n_train, n_val = len(X_train), len(X_val)
    X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
    # Forcing a dtype would turn an integer target into object, which
    # scikit-learn rejects as an "unknown" target.
    y = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    return Splits(X=X, y=y, folds=[(train_idx, val_idx)])


def cross_validation(X, y, folds: int, seed: int, task: str = CLASSIFICATION) -> Splits:
    """Classification stratifies when the labels allow and shuffles plainly
    otherwise, and regression always shuffles plainly. A fixed seed keeps the
    folds stable.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)          # dtype kept; see `holdout`

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


# Time-ordered splits validate only rows after every training row, so they never
# shuffle or stratify. Both expect rows sorted by time (`core.io` sorts them).


def time_holdout(X, y, test_size: float, gap: int = 0) -> Splits:
    """*gap* rows go unused, as labels a deployed model would still lack.
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
    """`time_cross_validation` validates *folds* successive blocks of the latest
    rows, and the first block is only trained on.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X, y = np.asarray(X), np.asarray(y)
    try:
        divided = list(TimeSeriesSplit(n_splits=folds, gap=int(gap)).split(X))
    except ValueError as exc:
        raise ValueError(f"{len(X)} rows cannot be divided into {folds} time-ordered "
                         f"folds with a gap of {gap}: {exc}") from exc
    return Splits(X=X, y=y, folds=divided)


def backtest(X, y, times, horizon: int, folds: int, gap: int = 0,
             series=None, season: int = 1) -> Splits:
    """`backtest` cuts at distinct *times*, so series sharing timestamps split
    at the same moment. Every fold uses the rows before its origin minus *gap*
    for training and validates the next *horizon* steps, only for series with
    earlier history.
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
