"""Datasets on disk and how one is read: the demo CSVs and mounted models an
instance offers, a dataset's separator, target and kinds of column, its rows
in time order, and the split a trial is evaluated over."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import columns, encoding, tasks
from .splits import MIN_FOLDS, backtest, time_cross_validation, time_holdout

#: The share of a dataset held out for validation when the scheme is a single
#: split, if the experiment does not say. Every experiment written before the
#: share was settable used this, so it is also what those files mean.
DEFAULT_TEST_SIZE = 0.2

_REPO_ROOT   = Path(__file__).parent.parent
_DATASETS_DIR = _REPO_ROOT / "datasets"
_MODELS_DIR   = _REPO_ROOT / "mounted_models"


def _files_in(directory: Path, pattern: str) -> dict[str, str]:
    """{stem: path} for the files in *directory* matching *pattern*, sorted."""
    if not directory.exists():
        return {}
    return {p.stem: str(p) for p in sorted(directory.glob(pattern))}


def demo_datasets() -> dict[str, str]:
    """Return {stem: path} for CSVs in datasets/.

    Files mounted to datasets/ at runtime (e.g. a Docker volume) appear in the
    dropdown automatically alongside the bundled demo CSVs.
    """
    return _files_in(_DATASETS_DIR, "*.csv")


def mounted_models() -> dict[str, str]:
    """Return {stem: path} for .py files in mounted_models/.

    Files mounted to mounted_models/ at runtime appear in the dropdown
    automatically.
    """
    return _files_in(_MODELS_DIR, "*.py")


def read_dataset(source):
    """A dataset as a DataFrame, from a path or from the file's bytes, with the
    separator sniffed.

    The create form has an upload in hand and should not have to write it out
    to look at it.
    """
    from io import BytesIO

    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(BytesIO(raw), sep=sep)


def target_of(source) -> tuple[str, Any]:
    """The target column of a dataset — its name and its values. *source* is a
    path or the file's bytes, as for `read_dataset`."""
    df = read_dataset(source)
    return str(df.columns[-1]), df.iloc[:, -1].to_numpy()


def orderable_columns(frame) -> list[str]:
    """The feature columns a dataset could be divided in time order by: dates,
    and numbers (a year, a period index). Dates first."""
    kinds = columns.infer_kinds(frame.iloc[:, :-1])
    return ([name for name, kind in kinds.items() if kind == columns.DATETIME]
            + [name for name, kind in kinds.items() if kind == columns.NUMERIC])


def forecast_problem(frame, column: str, series_column: str, horizon: int, folds: int,
                     gap: int):
    """Why *frame* cannot be backtested this way, or None if it can."""
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
    """Why *frame* cannot be divided in time order by *column* this way, or
    None if it can."""
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
    """*df* without the rows whose target is missing: there is nothing to train
    on in them and nothing to score them against."""
    return df[df.iloc[:, -1].notna()]


def dataset_encoding(csv_path: Path) -> dict:
    """How the dataset's feature columns become the floats a model is given —
    see `core.encoding`. A function of the file alone, so every caller that
    reads the same file encodes it the same way."""
    df = _labelled(read_dataset(csv_path))
    return encoding.fit(df.iloc[:, :-1])


def _in_time_order(df, column: str):
    """*df*'s rows sorted by *column*, oldest first, without the rows that have
    no time — a row that cannot be placed in time cannot be put on either side
    of a split in time.

    A date column is read as dates and a number column as numbers (a year, a
    period index). A column of labels has no order, and is refused.
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
    # Stable, so rows sharing a time keep the order the file gave them.
    order = np.argsort(times[known].to_numpy(), kind="stable")
    return df[known].iloc[order], times[known].iloc[order].to_numpy()


def load_frame(csv_path: Path, order_by: str | None = None):
    """The dataset as (X, y). The last column is the target.

    X is floats whatever the columns held: text and dates are encoded
    (`core.encoding`), a missing value is NaN. Rows without a target are left
    out. With *order_by*, the rows come oldest first by that column, and rows
    without a time are left out too.
    """
    df = _labelled(read_dataset(csv_path))
    # Fitted before any row is left out for having no time, so it is the same
    # encoding `dataset_encoding` gives and the models are told about.
    spec = encoding.fit(df.iloc[:, :-1])
    if order_by:
        df, _ = _in_time_order(df, order_by)
    return encoding.encode(spec, df.iloc[:, :-1]), df.iloc[:, -1].to_numpy()


def load_forecast(csv_path: Path, time_column: str, series_column: str = ""):
    """A forecast's rows, oldest first: (X, y, times, series). *series* is each
    row's series as text ("" when there is one series), *times* each row's time
    — the values a backtest is cut on."""
    df = _labelled(read_dataset(csv_path))
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
                 task: str = tasks.DEFAULT):
    """Reconstruct the identical train/val split from a CSV path, seed and
    validation share.

    A classification split is stratified when the classes allow it. A
    regression one never is: `train_test_split` would happily stratify a
    numeric target whose values each occur twice, treating every distinct
    number as a class.
    """
    X, y = load_frame(csv_path)
    size = float(test_size or DEFAULT_TEST_SIZE)
    if task == tasks.CLASSIFICATION:
        try:
            return train_test_split(X, y, test_size=size, random_state=seed, stratify=y)
        except ValueError:
            pass
    return train_test_split(X, y, test_size=size, random_state=seed)
