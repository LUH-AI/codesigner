"""What an experiment predicts: a class, or a number.

The task is chosen when an experiment is created and fixed for its life, like
how it is evaluated. It decides which models can be offered (a model says
which tasks it supports), which metrics a trial is scored with, and whether a
split may stratify on the target.

An experiment from before tasks existed is a classification one: every metric
and model there was then could only classify.
"""

from __future__ import annotations

import numpy as np

CLASSIFICATION = "classification"
REGRESSION = "regression"

#: Every task, in the order the form offers them.
TASKS = (CLASSIFICATION, REGRESSION)

#: What a model or a file that says nothing about its task means.
DEFAULT = CLASSIFICATION

#: An integer target with at most this many distinct values is guessed to be
#: classes rather than a quantity. Wine's `quality` (six values) is the case
#: this is for: it has always been tuned as classes here, and nothing about the
#: column says otherwise.
MAX_GUESSED_CLASSES = 20


def is_numeric(y) -> bool:
    """Whether every value of *y* is a number — what regression needs."""
    y = np.asarray(y)
    if y.dtype.kind in "biuf":
        return True
    try:
        np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def guess(y) -> str:
    """The task a target column most plausibly asks for.

    Text is classes. So is a handful of whole numbers. Anything else is a
    quantity. Only ever a default: the form shows it and the person creating
    the experiment can say otherwise.
    """
    y = np.asarray(y)
    if not is_numeric(y):
        return CLASSIFICATION
    values = np.asarray(y, dtype=float)
    values = values[~np.isnan(values)]
    whole = bool(np.all(np.mod(values, 1) == 0))
    if whole and len(np.unique(values)) <= MAX_GUESSED_CLASSES:
        return CLASSIFICATION
    return REGRESSION


def target_problem(y, task: str) -> str | None:
    """Why *y* cannot be the target of *task*, or None if it can."""
    if task == REGRESSION and not is_numeric(y):
        return "a regression target has to be numbers"
    return None


def supported(model) -> tuple[str, ...]:
    """The tasks *model* says it supports; a model that says nothing classifies."""
    tasks = getattr(model, "tasks", None)
    if isinstance(tasks, str):
        return (tasks,)
    return tuple(tasks) if tasks else (DEFAULT,)


def task_of(snapshot: dict) -> str:
    """The task a snapshot was made for."""
    task = (snapshot.get("evaluation") or {}).get("task")
    return task if task in TASKS else DEFAULT
