"""Which experiments' models can be exported, and the source to export them from.

Two kinds can: a built-in, from its module in `core/models/`, and an uploaded
model, from its file. A run imported from a SMAC directory never had a model
here, so there is nothing to export.
"""

import copy
import inspect
import re
import sys
from pathlib import Path

from django.utils.translation import gettext as _

from core import encoding, forecasting, io, processing, tasks
from core.model_export import export_model, export_upload
from core.model_source import inspect_model_source

from ..registry import MODELS


def unavailable(exp):
    """Why *exp*'s model cannot be exported, or None if it can.

    Reads nothing but the row, so the page can ask on every render.
    """
    data = exp.data
    if data.forecasts and not _columns(exp):
        return _("This forecast's dataset is not on this instance, and the exported "
                 "file needs to know how its columns were read.")
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


def build(exp, trial, metric, result=None):
    """The exported file's text for *trial* of *exp*: the tuned model, its
    hyperparameters and the experiment's data processing (see
    `core.model_export`)."""
    columns = _columns(exp)
    forecast = _forecast(exp, columns) if exp.data.forecasts else None
    about = _about(exp, trial, metric, result)
    if exp.data.model_file:
        text, class_name = source(exp)
        return export_upload(text, class_name=class_name, task=exp.data.task,
                             config=trial.config, seed=exp.data.seed, columns=columns,
                             about=about)
    model = _model(exp, columns, forecast)
    inner = getattr(model, "model", model)
    plan, _ = processing.resolve(exp.data.processing, inner)
    return export_model(model, config=trial.config, task=exp.data.task, seed=exp.data.seed,
                        columns=columns, plan=plan, about=about, forecast=forecast,
                        time_column=exp.data.time_column, series_column=exp.data.series_column)


def _model(exp, columns, forecast):
    """The built-in as this experiment ran it: told its task and its columns,
    and forecasting by reduction where the experiment forecasts and the model
    does not by itself."""
    entry = MODELS[exp.data.model_name]
    model = copy.copy(entry)
    model.task = exp.data.task
    model.feature_kinds = tuple(encoding.feature_kinds(columns)) if columns else ()
    model.processing = dict(exp.data.processing or {})
    if forecast is not None:
        model.forecast = dict(forecast)
        if not tasks.forecaster(entry):
            model = forecasting.Reduced(model, exp.data.task)
    return model


def _about(exp, trial, metric, result):
    """The exported file's docstring: what it predicts, and in two lines where
    its hyperparameters came from."""
    from ..views import _evaluation_label as evaluation_label

    target = ""
    dataset = exp.data.dataset_path()
    rows = None
    if dataset is not None and dataset.is_file():
        try:
            target, values = io.target_of(dataset)
            rows = len(values)
        except Exception:  # noqa: BLE001 — the file is still exportable without it
            pass
    task = {tasks.CLASSIFICATION: "classification", tasks.REGRESSION: "regression"}[exp.data.task]
    what = (f"{exp.data.model_name} forecasting {target or 'the target'}"
            if exp.data.forecasts else f"{exp.data.model_name} predicting {target or 'the target'}")
    score = trial.scores.get(metric)
    total = len(result.trials) if result is not None else None
    source_name = dataset.name if dataset is not None else "its dataset"
    lines = [
        f"{what} ({task}).",
        "",
        f"Tuned by Codesigner on {source_name}" + (f" ({rows} rows)" if rows else "") + ": "
        f"trial {trial.trial}" + (f" of {total}" if total else "")
        + f" by {exp.data.optimizer_name}"
        + (f", {metric} {score:.4g}" if score is not None else "")
        + f" ({evaluation_label(exp)}).",
        f'Experiment "{exp.title}" ({exp.identifier}).',
        "",
        "    uv run " + filename(exp, trial.trial)
        + (f" history.csv [--horizon {exp.data.horizon}] [--out forecast.csv]"
           if exp.data.forecasts else " train.csv --predict new.csv [--out predictions.csv]"),
    ]
    return "\n".join(lines)


def _forecast(exp, columns):
    """What the exported forecaster is told — the run's horizon, season and
    columns, the season inferred from the data when the experiment left it."""
    season = exp.data.season
    if not season:
        _, _, times, _ = io._load_forecast(exp.data.dataset_path(), exp.data.time_column,
                                           exp.data.series_column)
        season = forecasting.infer_season(times)
    return forecasting.context(columns, exp.data.time_column, exp.data.series_column,
                               exp.data.horizon, season)


def _columns(exp):
    """How *exp*'s dataset was encoded, for the file to encode its CSVs the
    same way — or None when the dataset is not on this instance, and the file
    falls back to the CSV's values as they are."""
    dataset = exp.data.dataset_path()
    if dataset is None or not dataset.is_file():
        return None
    return io.dataset_encoding(dataset)
