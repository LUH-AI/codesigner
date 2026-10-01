"""Which experiments' models can be exported, and the source to export them from.

Two kinds can: a built-in, from its module in `core/models/`, and an uploaded
model, from its file. A run imported from a SMAC directory never had a model
here, so there is nothing to export.
"""

import inspect
import re
import sys

from django.utils.translation import gettext as _

from core.model_export import export_model
from core.model_source import inspect_model_source

from ..registry import MODELS


def unavailable(exp):
    """Why *exp*'s model cannot be exported, or None if it can.

    Reads nothing but the row, so the page can ask on every render.
    """
    data = exp.data
    if data.model_file:
        if not data.model_file.storage.exists(data.model_file.name):
            return _("This experiment's model file is missing, so there is nothing to export.")
        return None
    if data.model_name in MODELS:
        return None
    return _("This experiment was imported without its model, so there is nothing to export.")


def source(exp):
    """(source text, class name) for *exp*'s model. Call `unavailable` first."""
    data = exp.data
    if data.model_file:
        with data.model_file.open("rb") as f:
            raw = f.read()
        info, error = inspect_model_source(raw)
        if info is None:
            raise ValueError(error)
        return raw.decode("utf-8"), info.class_name
    model = MODELS[data.model_name]
    return inspect.getsource(sys.modules[type(model).__module__]), type(model).__name__


def filename(exp, trial_n):
    """`random-forest-trial-15.py`: the model, then the trial it was tuned to."""
    slug = re.sub(r"[^a-z0-9]+", "-", exp.data.model_name.lower()).strip("-") or "model"
    return f"{slug}-trial-{trial_n}.py"


def build(exp, trial, metric):
    """The exported file's text for *trial* of *exp*."""
    text, class_name = source(exp)
    score = trial.scores.get(metric)
    description = (
        f"{exp.data.model_name}, exported from Codesigner.\n"
        f"Experiment: {exp.name} ({exp.identifier})\n"
        f"Trial {trial.trial}" + (f", {metric} {score:.6g}" if score is not None else "")
        + " on validation.\n\n"
        f"uv run {filename(exp, trial.trial)} train.csv --predict new.csv [--out predictions.csv] [--save model.pkl]")
    return export_model(text, class_name=class_name, task=exp.data.task, config=trial.config,
                        seed=exp.data.seed, description=description)
