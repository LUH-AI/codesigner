"""`save` writes a fitted model in its library's format, beside the fitted
processing that prepares new rows like the training rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from .jsonable import plain


class NothingToKeep(Exception):
    """`save` raises `NothingToKeep` for models without fitted parameters."""


def save_processing(processing, directory) -> list[str]:
    from sklearn.base import BaseEstimator
    from skops.io import dump

    directory = Path(directory)
    files = ["processing.json"]
    (directory / "processing.json").write_text(json.dumps(
        {"plan": processing.plan, "kinds": list(processing.kinds)}, indent=2))
    transformer = getattr(processing, "transformer_", None)
    if isinstance(transformer, BaseEstimator):
        dump(transformer, directory / "processing.skops")
        files.append("processing.skops")
    return files


def save(model, fitted, directory) -> list[str]:
    """`save` returns the files written, or raises `NothingToKeep`."""
    from sklearn.pipeline import Pipeline

    if not getattr(model, "has_parameters", True):
        raise NothingToKeep(f"{model.name} has no fitted parameters to keep")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    if isinstance(fitted, Pipeline) and "processing" in fitted.named_steps:
        files += save_processing(fitted.named_steps["processing"], directory)
        fitted = fitted.named_steps["model"]
    return files + list(model.save_parameters(fitted, directory))


def classes_of(fitted):
    """`classes_of` lists labels in probability order, or None for regression."""
    classes = getattr(fitted, "classes_", None)
    if classes is None:
        return None
    return [plain(value) for value in classes]
