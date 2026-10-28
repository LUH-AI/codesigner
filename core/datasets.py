"""`core.datasets` reads datasets and builds the split a trial is evaluated
over."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import columns, encoding, tasks
from .splits import MIN_FOLDS, backtest, time_cross_validation, time_holdout

#: Experiments that omit the share use this value.
DEFAULT_TEST_SIZE = 0.2

_REPO_ROOT   = Path(__file__).parent.parent
_DATASETS_DIR = _REPO_ROOT / "datasets"
_MODELS_DIR   = _REPO_ROOT / "mounted_models"


def _files_in(directory: Path, pattern: str) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {p.stem: str(p) for p in sorted(directory.glob(pattern))}


def demo_datasets() -> dict[str, str]:
    """Runtime mounts, such as a Docker volume, also appear in the dropdown.
    """
    return _files_in(_DATASETS_DIR, "*.csv")


def mounted_models() -> dict[str, str]:
    return _files_in(_MODELS_DIR, "*.py")


def target_last(df, target: str | None = None):
    """`target_last` moves the *target* column to the end, where every reader
    of a dataset looks for the target. A *target* of None leaves the last
    column as the target.
    """
    if not target:
        return df
    names = [str(c) for c in df.columns]
    if target not in names:
        raise ValueError(f"the dataset has no column named {target!r} to predict")
    at = names.index(target)
    return df[[c for i, c in enumerate(df.columns) if i != at] + [df.columns[at]]]


def read_dataset(source, target: str | None = None):
    """`read_dataset` sniffs the separator and accepts bytes, since the create
    form holds an upload in memory. The *target* column comes last.
    """
    from io import BytesIO

    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return target_last(pd.read_csv(BytesIO(raw), sep=sep), target)


def target_of(source, target: str | None = None) -> tuple[str, Any]:
    df = read_dataset(source, target)
    return str(df.columns[-1]), df.iloc[:, -1].to_numpy()


def orderable_columns(frame) -> list[str]:
    """Dates come first, then numbers such as a year."""
    kinds = columns.infer_kinds(frame.iloc[:, :-1])
    return ([name for name, kind in kinds.items() if kind == columns.DATETIME]
            + [name for name, kind in kinds.items() if kind == columns.NUMERIC])


def forecast_problem(frame, column: str, series_column: str, horizon: int, folds: int,
                     gap: int):
    """`forecast_problem` returns why backtesting fails, or None."""
    if horizon < 1:
        return "a forecast has to look at least one step ahead"
    try:
        ordered, times = _in_time_order(_labelled(frame), column)
        if series_column and series_column not in ordered.columns[:-1]:
            return f"the dataset has no feature column named {series_column!r}"
        if series_column == column:
            return "the series column cannot also be the time column"
        backtest(np.zeros((len(ordered), 1)), np.zeros(len(ordered)), times, horizon,
                 max(folds, 1), gap,
                 series=ordered[series_column].astype(str).to_numpy() if series_column else None)
    except ValueError as exc:
        return str(exc)
    return None


def time_order_problem(frame, column: str, folds: int, test_size: float, gap: int):
    """`time_order_problem` returns why the split fails, or None."""
    try:
        ordered, _ = _in_time_order(_labelled(frame), column)
        n = len(ordered)
        if folds >= MIN_FOLDS:
            time_cross_validation(np.zeros((n, 1)), np.zeros(n), folds, gap)
        else:
            time_holdout(np.zeros((n, 1)), np.zeros(n), test_size, gap)
    except ValueError as exc:
        return str(exc)
    return None


def _labelled(df):
    return df[df.iloc[:, -1].notna()]


def dataset_encoding(csv_path: Path, target: str | None = None) -> dict:
    """`dataset_encoding` depends on the file and its target alone, so every
    caller encodes it the same way."""
    df = _labelled(read_dataset(csv_path, target))
    return encoding.fit(df.iloc[:, :-1])


def _in_time_order(df, column: str):
    """`_in_time_order` drops rows without a time, since such rows fit neither
    side of a split, and refuses a label column.
    """
    if column not in df.columns[:-1]:
        raise ValueError(f"the dataset has no feature column named {column!r} to order it by")
    kind = columns.kind_of(df[column])
    if kind == columns.CATEGORICAL:
        raise ValueError(f"{column!r} holds labels, which have no order in time")
    if kind == columns.DATETIME:
        times = pd.to_datetime(df[column], errors="coerce", format="mixed", utc=True)
    else:
        times = pd.to_numeric(df[column], errors="coerce")
    known = times.notna()
    # Stable, so tied times keep file order.
    order = np.argsort(times[known].to_numpy(), kind="stable")
    return df[known].iloc[order], times[known].iloc[order].to_numpy()


def load_frame(csv_path: Path, order_by: str | None = None, target: str | None = None):
    """`load_frame` encodes X as floats and drops rows without a target, or,
    with *order_by*, without a time.
    """
    df = _labelled(read_dataset(csv_path, target))
    # Fitted before time filtering, to match `dataset_encoding`.
    spec = encoding.fit(df.iloc[:, :-1])
    if order_by:
        df, _ = _in_time_order(df, order_by)
    return encoding.encode(spec, df.iloc[:, :-1]), df.iloc[:, -1].to_numpy()


def load_forecast(csv_path: Path, time_column: str, series_column: str = "",
                  target: str | None = None):
    """*series* is each row's series as text, or None without a series column;
    *times* is what a backtest cuts on."""
    df = _labelled(read_dataset(csv_path, target))
    spec = encoding.fit(df.iloc[:, :-1])
    df, times = _in_time_order(df, time_column)
    series = None
    if series_column:
        if series_column not in df.columns[:-1]:
            raise ValueError(f"the dataset has no feature column named {series_column!r} "
                             f"to tell its series apart by")
        series = df[series_column].astype(str).to_numpy()
    return encoding.encode(spec, df.iloc[:, :-1]), df.iloc[:, -1].to_numpy(), times, series


def load_splits(csv_path: Path, seed: int, test_size: float = DEFAULT_TEST_SIZE,
                task: str = tasks.DEFAULT, target: str | None = None):
    """`load_splits` stratifies classification when possible; `train_test_split`
    would treat repeated regression values as classes.
    """
    X, y = load_frame(csv_path, target=target)
    size = float(test_size or DEFAULT_TEST_SIZE)
    if task == tasks.CLASSIFICATION:
        try:
            return train_test_split(X, y, test_size=size, random_state=seed, stratify=y)
        except ValueError:
            pass
    return train_test_split(X, y, test_size=size, random_state=seed)
