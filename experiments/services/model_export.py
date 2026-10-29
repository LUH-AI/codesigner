"""`model_export` exports a built-in model from `core/models/` or an uploaded
model from its file; a SMAC import has none.
"""

import copy
import inspect
import re
import sys

from django.utils.translation import gettext as _
from django.utils.translation import ngettext

from core import datasets, encoding, forecasting, processing, tasks
from core.model_export import export_model, export_upload
from core.model_source import inspect_model_source
from core.registry import MODELS


def unavailable(exp):
    """The check only reads the row, so a page may call it on each render.
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
    """`source` returns code and class after `unavailable`."""
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
    slug = re.sub(r"[^a-z0-9]+", "-", exp.data.model_name.lower()).strip("-") or "model"
    return f"{slug}-trial-{trial_n}.py"


def build(exp, trial, metric, result=None):
    """`build` exports the tuned model, hyperparameters and data processing."""
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
    """`_model` sets up the built-in with the experiment's task and columns, and
    `Reduced` wraps a non-forecaster when the experiment forecasts."""
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


def evaluation_label(exp):
    """`evaluation_label` marks a SMAC import as unrecorded because its split
    columns contain placeholder defaults, and reporting them would invent a
    holdout.
    """
    if exp.data.model_name not in MODELS and not exp.data.model_file:
        return _("Not recorded")
    if exp.data.forecasts:
        return ngettext("%(k)s backtest of %(h)s steps by %(column)s",
                        "%(k)s backtests of %(h)s steps by %(column)s",
                        max(exp.data.cv_folds, 1)) % {
            "k": max(exp.data.cv_folds, 1), "h": exp.data.horizon, "column": exp.data.time_column}
    if exp.data.time_column:
        if exp.data.cv_folds >= 2:
            return _("%(k)s time-ordered folds by %(column)s") % {
                "k": exp.data.cv_folds, "column": exp.data.time_column}
        return _("latest %(pct)s%% by %(column)s held out") % {
            "pct": round(exp.data.test_size * 100), "column": exp.data.time_column}
    if exp.data.cv_folds >= 2:
        return _("%(k)s-fold CV") % {"k": exp.data.cv_folds}
    return _("%(pct)s%% held out") % {"pct": round(exp.data.test_size * 100)}


def _about(exp, trial, metric, result):
    target = ""
    dataset = exp.data.dataset_path()
    rows = None
    if dataset is not None and dataset.is_file():
        try:
            target, values = datasets.target_of(dataset)
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
    """Season falls back to inference from the data."""
    season = exp.data.season
    if not season:
        _, _, times, _ = datasets.load_forecast(exp.data.dataset_path(), exp.data.time_column,
                                           exp.data.series_column)
        season = forecasting.infer_season(times)
    return forecasting.context(columns, exp.data.time_column, exp.data.series_column,
                               exp.data.horizon, season)


def _columns(exp):
    """`_columns` returns None without the dataset, and the file then keeps raw
    CSV values."""
    dataset = exp.data.dataset_path()
    if dataset is None or not dataset.is_file():
        return None
    return datasets.dataset_encoding(dataset)
