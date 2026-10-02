"""A fitted trial model's parameters, written out in its library's own format.

What a trial fitted is the experiment's processing in front of the model (see
`core.models.base.Tunable.fit`). Both are written: the model by the model,
which knows its library's format (`OutputBaseModel.save_parameters`), and the
processing's fitted state — medians, scale, the labels it saw — beside it, so
new rows can be prepared exactly as the training rows were.
"""

from __future__ import annotations

import json
from pathlib import Path


class NothingToKeep(Exception):
    """The model has no fitted parameters — its prediction is its history."""


def save_processing(processing, directory) -> list[str]:
    """*processing*'s fitted state: ``processing.json`` (the plan and the kind
    of each column it was given) and, when it learned anything,
    ``processing.skops`` (the fitted scikit-learn transformer)."""
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
    """Write *fitted* — what *model* fitted in a trial — into *directory*,
    and return the files written. Raises `NothingToKeep` for a model with
    nothing fitted to keep."""
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
    """A fitted classifier's labels, in the order its probabilities and codes
    use, as plain values — or None for a regression."""
    classes = getattr(fitted, "classes_", None)
    if classes is None:
        return None
    return [value.item() if hasattr(value, "item") else value for value in classes]
