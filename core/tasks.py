"""`tasks` defines what an experiment predicts, a class or a number, fixed at
creation.

The task picks the models offered, the metrics, and whether a split may stratify
on the target.
"""

from __future__ import annotations

import numpy as np

CLASSIFICATION = "classification"
REGRESSION = "regression"

#: Form order; forecasting is an evaluation mode (`core.forecasting`).
TASKS = (CLASSIFICATION, REGRESSION)

#: Task for a model or file naming none.
DEFAULT = CLASSIFICATION


def is_numeric(y) -> bool:
    y = np.asarray(y)
    if y.dtype.kind in "biuf":
        return True
    try:
        np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def target_problem(y, task: str) -> str | None:
    """Reason *y* cannot serve *task*, or None."""
    if task == REGRESSION and not is_numeric(y):
        return "a regression target has to be numbers"
    return None


def supported(model) -> tuple[str, ...]:
    """`supported` defaults a model without tasks to classification."""
    tasks = getattr(model, "tasks", None)
    if isinstance(tasks, str):
        return (tasks,)
    return tuple(tasks) if tasks else (DEFAULT,)


def forecaster(model) -> bool:
    """`forecaster` tells whether a model forecasts by itself
    (`core.forecasting`)."""
    return bool(getattr(model, "forecaster", False))


def task_of(snapshot: dict) -> str:
    task = (snapshot.get("evaluation") or {}).get("task")
    return task if task in TASKS else DEFAULT
