"""What the assistant is told about a dataset: worked out here, never by the
model.

The model gets facts it could not make up — how many rows, each column's kind
as Codesigner reads it, how much of it is missing, how many distinct values it
has — and a few example values of each, as many as `ASSISTANT_SAMPLE_ROWS`
allows. With that at 0, nothing of the data itself leaves this machine: only
names, kinds and counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings

from core import columns as kinds
from core import datasets, tasks

#: More distinct values than this, in a target of whole numbers, reads as a
#: quantity rather than as classes.
_FEW_CLASSES = 20


def profile(path: Path, target: str = "") -> dict:
    """The facts about the dataset at *path*, *target* the column predicted
    (the last when "")."""
    frame = datasets.read_dataset(path, target or None)
    samples = max(0, int(settings.ASSISTANT_SAMPLE_ROWS))
    feature_kinds = kinds.infer_kinds(frame.iloc[:, :-1])
    target_name = str(frame.columns[-1])
    out = {
        "rows": int(len(frame)),
        "columns": [],
        "target": _target(frame.iloc[:, -1], target_name, samples, chosen=bool(target)),
        "time_order_candidates": datasets.orderable_columns(frame),
    }
    for name in frame.columns[:-1]:
        out["columns"].append(_column(frame[name], str(name), feature_kinds.get(name), samples))
    return out


def _column(values: pd.Series, name: str, kind: str, samples: int) -> dict:
    present = values.dropna()
    column = {
        "name": name,
        "kind": kind,
        "missing": int(values.isna().sum()),
        "distinct": int(present.nunique()),
    }
    if column["distinct"] == len(values) and len(values) > _FEW_CLASSES:
        column["note"] = "every value is different: an identifier, most likely"
    elif column["distinct"] <= 1:
        column["note"] = "one value throughout: tells a model nothing"
    if kind == kinds.NUMERIC and len(present):
        numbers = pd.to_numeric(present, errors="coerce").dropna()
        if len(numbers):
            column["min"] = _plain(numbers.min())
            column["max"] = _plain(numbers.max())
            column["mean"] = round(float(numbers.mean()), 4)
    if samples:
        column["examples"] = [_plain(v) for v in present.drop_duplicates().head(samples)]
    return column


def _target(values: pd.Series, name: str, samples: int, chosen: bool) -> dict:
    """The predicted column, and the task it reads as — a suggestion for the
    person to confirm, never a decision: a column of whole numbers is classes
    to one person and a quantity to the next."""
    present = values.dropna()
    numeric = tasks.is_numeric(present.to_numpy())
    distinct = int(present.nunique())
    whole = numeric and bool(np.all(np.mod(pd.to_numeric(present), 1) == 0)) if len(present) else False
    if not numeric:
        suggestion = tasks.CLASSIFICATION
    elif whole and distinct <= _FEW_CLASSES:
        suggestion = "either"
    else:
        suggestion = tasks.REGRESSION
    out = {
        "name": name,
        "chosen": chosen,
        "missing": int(values.isna().sum()),
        "distinct": distinct,
        "numeric": numeric,
        "suggested_task": suggestion,
    }
    if distinct <= _FEW_CLASSES:
        counts = present.value_counts()
        out["smallest_class_share"] = round(float(counts.min() / counts.sum()), 4) if len(counts) else None
        if samples:
            out["classes"] = {str(k): int(v) for k, v in counts.head(_FEW_CLASSES).items()}
    elif numeric and len(present):
        out["min"] = _plain(present.min())
        out["max"] = _plain(present.max())
    return out


def _plain(value):
    """A value as JSON has it."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
