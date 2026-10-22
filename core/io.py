"""The `.ihpo` experiment file, and building a runnable experiment from one.

File format (.ihpo): UTF-8 JSON. One object per subject, each stating its
subject once, and `result` last because it is the longest by two orders of
magnitude:

    format      int   — the shape of this file (see SNAPSHOT_FORMAT)
    version     str   — the application version that wrote it
    name        str
    seed        int   — one seed for the whole experiment: the split, the
                        sampler, the scenario, fit_predict, the importance
    dataset     dict  — path, filename, and (on export) digest and shape
    model       dict  — kind, name, path, and (on export) digest and lock
    evaluation  dict  — scheme, folds, test_size, and (on export) stratified
    metrics     dict  — names, current, original
    optimizer   dict  — name, params, and (on export) defaults_used. For a run
                        imported from a SMAC directory, `params` is read back
                        out of `scenario.json`'s `_meta` (see
                        `smac_import._optimizer_settings`), so an imported run
                        says how it was configured rather than coming back as a
                        bare name with defaults underneath it.
    space       dict | absent — the search space, as ConfigSpace's own
                        serialized dict; byte-identical to SMAC's
                        configspace.json. Optional: a file without one falls
                        back to asking the model, which is what every file
                        written before this section existed does.
    priors      dict | absent — beliefs stated about the search space, by
                        hyperparameter: `kind`, `params`, `decay`
                        (`{shape, beta}`) and `at_trial`, the trial the belief
                        was stated at. These steer the *next* run rather than
                        describing a past one, which is why they are here and
                        not under `runs`. Absent when nothing was stated.
    runs        list  — export only: the history of runs over the trials
    environment dict  — export only: the versions behind the numbers
    result      dict | null — serialized OptimizationResult, a strict superset
                        of SMAC's runhistory.json (see serialize_result)

The superset property is about `result`, not about the file. A runhistory has
none of `name`, `seed`, `dataset.path`, `model.name` or `metrics.names`, all of
which `parse` requires — so a runhistory is not an .ihpo, while every .ihpo's
`result` contains everything a runhistory does. `smac_import` is what bridges
the two, filling the rest from the directory around the runhistory.

The dataset arrays and model object are NOT stored. On load the two paths are
resolved from disk; missing files must be re-supplied by the caller (or the
experiment loaded read-only).

Every reader goes through :func:`normalize`, so an older file — which spelled
all of this as one flat namespace — is lifted rather than refused.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import encoding, forecasting, tasks
from .datasets import (
    DEFAULT_TEST_SIZE, dataset_encoding, load_forecast, load_frame, load_splits,
)
from .jsonable import finite
from .metrics import declarations
from .models.base import Tunable
from .paths import MAX_STATE_FILES, is_safe_relative
from .processing import choices_of
from .registry import optimizer_class
from .splits import (
    MIN_FOLDS, backtest, cross_validation, holdout, time_cross_validation, time_holdout,
)

# ── Dataset / model discovery ─────────────────────────────────────────────────

# ── Custom model loading ──────────────────────────────────────────────────────

def load_model_from_path(model_path: str) -> tuple:
    """Load a BaseModel subclass from a .py file at *model_path*.

    Executes the file as Python — callers are responsible for only passing
    trusted paths (see the ALLOW_CUSTOM_MODELS setting).

    Returns ``(instance, None)`` on success or ``(None, error_str)`` on failure.
    Error strings are plain English and not translated.
    """
    from codesigner_model import BaseModel

    path = Path(model_path)
    if not path.is_file():
        return None, f"File not found: {model_path}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as f:
        f.write(path.read_bytes())
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_custom_model", tmp_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return None, f"Could not load file: {e}"

        candidates = [
            obj for obj in vars(module).values()
            if isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
        ]

        if not candidates:
            return None, "No BaseModel subclass found in the file."

        for cls in candidates:
            try:
                return cls(), None
            except TypeError:
                abstract = sorted(
                    name for name, val in inspect.getmembers(cls)
                    if getattr(val, "__isabstractmethod__", False)
                )
                return None, f"'{cls.__name__}' is missing required methods: {', '.join(abstract)}."
            except Exception as e:
                return None, f"Could not instantiate '{cls.__name__}': {e}"

        return None, "No valid BaseModel subclass could be instantiated."
    finally:
        os.unlink(tmp_path)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars / arrays to plain Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



# ── Public API ────────────────────────────────────────────────────────────────

#: The shape of the file. Format 1 was one flat namespace, and the sections
#: describing how an experiment was made were added beside it — so the optimizer
#: was named twice, the fold count lived under two names, and every setting
#: appeared once as asked for and again as resolved. Format 2 gives each subject
#: one object and states it once. A file without the key is format 1.
SNAPSHOT_FORMAT = 2


def _check_format(snapshot: dict) -> None:
    """Refuse a file written in a format newer than this build understands.

    Only formats *older* than the current one can be lifted. `normalize`
    dispatched on exact equality, so anything that was not the current format
    fell through to the format 1 lifter — which does not fail on a future file.
    It quietly returns a snapshot with most of its sections empty, and the
    failure then surfaces several screens later as something else entirely.
    """
    written_as = snapshot.get("format")
    if isinstance(written_as, int) and written_as > SNAPSHOT_FORMAT:
        raise ValueError(
            f"this file is in format {written_as}, which is newer than the "
            f"{SNAPSHOT_FORMAT} this version of codesigner understands")


def normalize(snapshot: dict) -> dict:
    """A snapshot of either format, in the current one.

    Everything that reads a snapshot calls this, so the rest of the application
    knows one shape. A format 1 file is lifted rather than refused: those files
    are on other people's disks and there is nothing wrong with them.

    The record-only parts of a format 1 file are dropped rather than lifted —
    its `optimizer.resolved`, and the copies of the settings that its `runs[]`
    entries carried. Nothing reconstructs from them, and an export rebuilds the
    record from the experiment as it stands.
    """
    if snapshot.get("format") == SNAPSHOT_FORMAT:
        return snapshot

    _check_format(snapshot)

    model_path = snapshot.get("model_path", "")
    recorded_model = snapshot.get("model") or {}
    lifted = {
        "format": SNAPSHOT_FORMAT,
        "version": snapshot.get("version"),
        "name": snapshot.get("name"),
        "seed": snapshot.get("seed"),
        "dataset": {**(snapshot.get("data") or {}),
                    "path": snapshot.get("dataset_path", "")},
        "model": {**recorded_model,
                  "kind": "file" if model_path else "registry",
                  "name": snapshot.get("model_name"),
                  "path": model_path},
        "evaluation": _lifted_evaluation(snapshot),
        "metrics": {"names": snapshot.get("metric_names") or [],
                    "current": snapshot.get("primary_metric"),
                    "original": snapshot.get("original_metric")},
        "optimizer": {"name": snapshot.get("optimizer_name"),
                      "params": snapshot.get("optimizer_params") or {}},
    }
    for section in ("space", "priors", "runs", "environment", "began_at"):
        if snapshot.get(section) is not None:
            lifted[section] = snapshot[section]
    lifted["result"] = snapshot.get("result")
    return lifted


def _lifted_evaluation(snapshot: dict) -> dict:
    """Format 1's `cv_folds`, plus whatever its `evaluation` section resolved.

    `scheme`, `folds` and `test_size` are all derivable from `cv_folds`, which
    is why they are not carried separately any more. `stratified` is not — it is
    what scikit-learn agreed to do with that target, not what was asked for.
    """
    folds = int(snapshot.get("cv_folds") or 0)
    kfold = folds >= MIN_FOLDS
    recorded = snapshot.get("evaluation") or {}
    return {
        "scheme": "kfold" if kfold else "holdout",
        "folds": folds if kfold else None,
        "test_size": None if kfold else 0.2,
        "stratified": recorded.get("stratified"),
    }


def evaluation_of(snapshot: dict) -> dict:
    """How this snapshot's rows are divided for evaluation, every setting
    present: `folds` (0 is a single holdout), `test_size` (the holdout's share;
    under cross-validation `DEFAULT_TEST_SIZE`, so a later switch of scheme has
    something to start from), `time_column` ("" for a division that ignores
    time), `gap` (rows left between training and validation), and a
    forecast's `horizon`, `series_column` and `season` (zeros and "" for a
    snapshot not evaluated by backtests: a horizon above zero is what makes a
    forecast)."""
    section = snapshot.get("evaluation") or {}
    return {
        "folds": int(section.get("folds") or 0),
        "test_size": float(section.get("test_size") or DEFAULT_TEST_SIZE),
        "time_column": str(section.get("time_column") or ""),
        "gap": int(section.get("gap") or 0),
        "horizon": int(section.get("horizon") or 0),
        "series_column": str(section.get("series_column") or ""),
        "season": int(section.get("season") or 0),
    }


def to_bytes(snapshot: dict) -> bytes:
    """*snapshot* as the bytes of an `.ihpo` file.

    The one place a snapshot becomes a file, so `save` and the export view
    cannot disagree about what one may contain — see `finite`.
    """
    return json.dumps(finite(snapshot), ensure_ascii=False, indent=2,
                      default=_json_default).encode("utf-8")


def parse(data: bytes) -> dict:
    """Decode and validate raw bytes; return the snapshot dict, normalized.

    Raises ``ValueError`` with a human-readable message on any problem.
    Does NOT reconstruct any objects — call :func:`build_experiment` for that.
    """
    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("version"), str):
        raise ValueError("not a valid .ihpo file or unsupported version")

    # Before the field checks below, which are chosen by format: asked of a
    # newer file they report whichever format 1 field it happens to lack, which
    # names a consequence rather than the problem.
    _check_format(raw)

    # Checked before normalizing, against the names the file itself uses: a
    # field missing from a format 1 file should be reported by the name it is
    # missing under, not by whatever normalizing would have called it.
    if raw.get("format") == SNAPSHOT_FORMAT:
        missing = [f"{section}.{key}"
                   for section, key in (("dataset", "path"), ("model", "name"),
                                        ("optimizer", "name"), ("optimizer", "params"),
                                        ("metrics", "names"))
                   if not isinstance(raw.get(section), dict) or key not in raw[section]]
    else:
        missing = [key for key in ("model_name", "optimizer_name", "optimizer_params",
                                   "metric_names", "dataset_path")
                   if key not in raw]
    missing += [key for key in ("name", "seed") if key not in raw]
    if missing:
        raise ValueError(f"missing field: {missing[0]!r}")

    snapshot = normalize(raw)
    check_optimizer_state(snapshot)
    _check_trial_scores(snapshot)
    return snapshot


def check_optimizer_state(snapshot: dict) -> None:
    """Refuse a result whose optimizer files name paths of their own choosing.

    `optimizer_state` is keyed by the relative path each file was written to,
    and deserializing writes them back out. Checking here means a file edited to
    point somewhere else is refused when it is read, rather than at whatever
    later moment something happens to rebuild its result.
    """
    state = (snapshot.get("result") or {}).get("optimizer_state")
    if state is None:
        return
    if not isinstance(state, dict):
        raise ValueError("optimizer_state must be an object")
    if len(state) > MAX_STATE_FILES:
        raise ValueError(f"optimizer_state has more than {MAX_STATE_FILES} entries")
    for name in state:
        if not is_safe_relative(name):
            raise ValueError(f"unsafe path in experiment file: {name!r}")


def _check_trial_scores(snapshot: dict) -> None:
    """Refuse a result whose trials don't carry every metric the file declares.

    Every figure reads `trial.scores[metric]` for whichever metric is being
    viewed, and the metric selector offers whatever `metrics.names` lists — so a
    trial missing one of them is not a degraded figure, it is a KeyError at
    render time, i.e. a 500 on the detail page for a file that imported
    cleanly. Refusing it here means the failure lands where it can be
    explained, naming the trial and the metric, instead of several screens
    later with no indication which file was at fault.

    Mirrors `BaseOptimizer.deserialize_result`'s own fallback so the two agree
    on what a valid entry is: an entry with no `scores` at all is read as
    carrying the primary metric alone (that is how single-metric files written
    before per-metric scores existed are still loadable), and is therefore fine
    exactly when the file declares no other metric.
    """
    result = snapshot.get("result") or {}
    names = list(snapshot.get("metrics", {}).get("names") or [])
    if not names:
        return
    primary = result.get("primary_metric") or ""
    for entry in result.get("data") or []:
        scores = entry.get("scores")
        if scores is None:
            scores = {primary: None} if primary else {}
        missing = [name for name in names if name not in scores]
        if missing:
            trial = entry.get("config_id", "?")
            raise ValueError(
                f"trial {trial} has no score for {missing[0]!r}, which this "
                f"experiment lists as one of its metrics")


def dataset_path_ok(snapshot: dict) -> bool:
    """Return True if the stored dataset path exists and is a readable file."""
    p = (normalize(snapshot).get("dataset") or {}).get("path", "")
    return bool(p) and Path(p).is_file()


def model_path_ok(snapshot: dict) -> bool:
    """Return True if the model is a registry model (no path) or its file exists."""
    p = (normalize(snapshot).get("model") or {}).get("path", "")
    return not p or Path(p).is_file()


def attach_dataset(exp: dict, dataset_path: str) -> None:
    """Load *dataset_path*, reproduce the stored train/val split, and update *exp* in-place.

    Raises ``ValueError`` if the file cannot be found or read.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"dataset not found: {dataset_path}")
    X_train, X_val, y_train, y_val = load_splits(
        path, exp["seed"], exp.get("test_size") or DEFAULT_TEST_SIZE,
        exp.get("task", tasks.DEFAULT))
    exp["X_train"]      = X_train
    exp["y_train"]      = y_train
    exp["X_val"]        = X_val
    exp["y_val"]        = y_val
    exp["dataset_path"] = str(path.resolve())


def build_experiment(
    snapshot: dict,
    available_metrics: dict,
    available_models: dict,
    available_optimizers: dict,
    model_name: str | None = None,
    read_only: bool = False,
    load_model: bool = True,
) -> tuple[str, dict]:
    """Construct a live experiment dict from a parsed snapshot.

    Parameters
    ----------
    snapshot:
        Output of :func:`parse`.
    available_metrics / available_models / available_optimizers:
        The registries from the running app.
    model_name:
        Override the registry key when the stored built-in model name is missing.
        Ignored when ``snapshot["model_path"]`` is non-empty.
    read_only:
        If True, skip all file-based loading: dataset arrays are None and any
        custom model file is not loaded (model will be None).  Registry-based
        models are still resolved normally since they require no file I/O.
    load_model:
        If False, a custom model file is not imported and ``model`` is None,
        while the dataset is still loaded.  For a caller that runs the model in
        its own process and only needs everything around it.

    Returns ``(name, exp_dict)`` on success.  Raises ``ValueError`` on failure.
    """
    snapshot = normalize(snapshot)
    ev = evaluation_of(snapshot)
    metric_names = snapshot["metrics"]["names"]
    model_path = snapshot["model"].get("path", "")
    task = tasks.task_of(snapshot)

    # A file may bring its own metric. A run imported from somewhere else names
    # an objective this build has never heard of, and refusing it would mean the
    # file cannot be opened at all — so the result's own declarations are
    # consulted alongside the registry. They are enough to read, compare and
    # draw the numbers in the file, which is everything except computing new
    # ones; `undescribe` makes that explicit by refusing to score.
    declared = declarations((snapshot.get("result") or {}).get("declared_metrics"))
    missing_metrics = [m for m in metric_names
                       if m not in available_metrics and m not in declared]
    if missing_metrics:
        raise ValueError(f"unknown metric(s): {', '.join(missing_metrics)}")

    opt_name = snapshot["optimizer"]["name"]
    # Aliases too: an optimizer that has been renamed still has to answer to
    # what it was called in every .ihpo written before the rename.
    opt_type = optimizer_class(opt_name, available_optimizers)
    if opt_type is None:
        raise ValueError(f"optimizer '{opt_name}' is not available")

    seed = snapshot["seed"]

    # ── Model ─────────────────────────────────────────────────────────────────
    if snapshot["model"].get("kind") == "external":
        # A run somebody else produced. There is no model to resolve and no
        # dataset to run it against — the file records what was searched and
        # what it scored, not what was doing the searching. `model = None` is
        # the state a custom-model experiment viewed read-only is already in, so
        # nothing downstream needs a new branch; `_model_available` reads it as
        # not runnable, which is the whole of what "read-only" means here.
        model = None
        resolved_model = snapshot["model"].get("name") or "External"
    elif model_path:
        if read_only or not load_model:
            # `load_model=False` means the caller will supply the model itself —
            # a custom model usually runs in its own process now, and importing
            # it here to then throw it away is exactly what that avoids.
            model = None
        else:
            model, err = load_model_from_path(model_path)
            if err:
                raise ValueError(f"custom model error: {err}")
            give_context(model, "task", task)
        resolved_model = snapshot["model"]["name"]
    else:
        stored_model = model_name if model_name is not None else snapshot["model"]["name"]
        model_entry = (
            available_models.get(stored_model)
            or next((m for m in available_models.values()
                     if m.name == stored_model or stored_model in getattr(m, "aliases", ())),
                    None)
        )
        if model_entry is None:
            raise ValueError(f"model '{stored_model}' is not available")
        if task not in tasks.supported(model_entry):
            raise ValueError(f"model '{stored_model}' does not do {task}")
        forecasts = ev["horizon"] > 0
        if tasks.forecaster(model_entry) and not forecasts:
            raise ValueError(f"model '{stored_model}' only forecasts")
        # A copy, told its task: the registry's instance is shared by every
        # experiment in the process, and the task is this one's. A model asked
        # to forecast that does not by itself does so through `core.forecasting`.
        if forecasts and not tasks.forecaster(model_entry):
            model = forecasting.Reduced(model_entry, task)
        else:
            model = copy.copy(model_entry)
            model.task = task
        resolved_model = stored_model

    # ── Dataset ───────────────────────────────────────────────────────────────
    # `splits` is what a trial is actually evaluated over — one fold for a
    # holdout, k for cross-validation. The four arrays stay as they were: plenty
    # of callers read them, and for a holdout they are the same data.
    cv_folds, test_size = ev["folds"], ev["test_size"]
    forecast = times = None
    if read_only:
        X_train = X_val = y_train = y_val = None
        splits = None
        columns = None
    else:
        dataset_path = snapshot["dataset"].get("path", "")
        path = Path(dataset_path)
        if not path.is_file():
            raise ValueError(f"dataset not found: {dataset_path}")
        columns = dataset_encoding(path)
        if model is not None:
            give_context(model, "feature_kinds", tuple(encoding.feature_kinds(columns)))
            # A built-in is told how its columns are processed, and so is one
            # wrapped for forecasting; an uploaded model is given them as they are.
            processing = choices_of((snapshot.get("dataset") or {}).get("processing"))
            for told in (getattr(model, "model", None), model):
                if isinstance(told, Tunable):
                    told.processing = dict(processing)
        time_column, gap = ev["time_column"], ev["gap"]
        if ev["horizon"] > 0:
            if not time_column:
                raise ValueError("a forecast needs the column its rows are ordered in time by")
            shape = ev
            X_all, y_all, times, series = load_forecast(path, time_column, shape["series_column"])
            season = shape["season"] or forecasting.infer_season(times)
            splits = backtest(X_all, y_all, times, shape["horizon"], max(cv_folds, 1), gap,
                              series=series, season=season)
            forecast = forecasting.context(columns, time_column, shape["series_column"],
                                           shape["horizon"], season, gap)
            if model is not None:
                give_context(model, "forecast", dict(forecast))
            # The latest backtest, as the four arrays: the one nearest to how
            # the forecast would be used.
            train_idx, val_idx = splits.folds[-1]
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
        elif time_column:
            # In time order: never shuffled, never stratified, the seed has no
            # say in it. The holdout's four arrays are its one fold, so callers
            # reading them see the same division a trial does.
            X_all, y_all = load_frame(path, order_by=time_column)
            splits = (time_cross_validation(X_all, y_all, cv_folds, gap) if cv_folds >= MIN_FOLDS
                      else time_holdout(X_all, y_all, test_size, gap))
            train_idx, val_idx = splits.folds[0]
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
        else:
            X_train, X_val, y_train, y_val = load_splits(path, seed, test_size, task)
            if cv_folds >= MIN_FOLDS:
                X_all, y_all = load_frame(path)
                splits = cross_validation(X_all, y_all, cv_folds, seed, task)
            else:
                splits = holdout(X_train, y_train, X_val, y_val)

    # The registry first: a later build correcting what a metric means must not
    # be overruled by a file written before the correction.
    metrics   = {m: available_metrics.get(m) or declared[m] for m in metric_names}
    # Narrowed to what this optimizer still accepts. A withdrawn or renamed
    # parameter would otherwise be an unexpected keyword, raised from whichever
    # page happened to rebuild the result.
    optimizer = opt_type(**opt_type.known_params(snapshot["optimizer"].get("params") or {}))
    result    = optimizer.deserialize_result(snapshot["result"]) if snapshot.get("result") else None

    exp = {
        "model":           model,
        "model_name":      resolved_model,
        "model_path":      model_path,
        "optimizer":       optimizer,
        "current_metric":  snapshot["metrics"].get("current"),
        "original_metric": snapshot["metrics"].get("original"),
        "metrics":         metrics,
        "seed":            seed,
        "dataset_path":    snapshot["dataset"].get("path", ""),
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "cv_folds": cv_folds,
        "test_size": test_size,
        "time_column": ev["time_column"],
        "gap": ev["gap"],
        "task":    task,
        "splits":  splits,
        # How the features were encoded (`core.encoding`), and so what kind each
        # column of X is. None when nothing was read.
        "encoding": columns,
        "feature_kinds": encoding.feature_kinds(columns) if columns else None,
        # What a forecaster is told (`core.forecasting`), None for any other run.
        "forecast": forecast,
        # Each row of the splits' time, in their order; a forecast's only.
        "times": times,
        # Undecoded. Kept as the serialized dict the file carries, because the
        # one caller that wants a live object also wants it seeded, and
        # `config_space_from_serialized` does both.
        "config_space": snapshot.get("space"),
        "result":  result,
    }
    return snapshot["name"], exp


def give_context(model, attribute: str, value) -> None:
    """Set *attribute* on *model*: its task, the kind of each of its columns, or
    its forecast. A model written before the attribute existed may not take it
    (a property with no setter, say) and is left as it is: it was written for
    what it could be offered then, a classification over numeric columns."""
    try:
        setattr(model, attribute, value)
    except AttributeError:
        pass


def config_space_from_serialized(space: dict | None, seed: int = 0):
    """A stored `space` section as a live `ConfigurationSpace`, or None.

    Copied before decoding because `from_serialized_dict` *consumes* what it is
    given — it pops each hyperparameter's `type` — so a second call on the same
    dict raises, and the dict here belongs to a snapshot the caller keeps using.
    The same rule `core/modelhost/client.py`'s `get_config_space` follows, for
    the same reason.

    Serializing loses the seed, and `RandomOptimizer` samples from this, so it
    is re-applied rather than inherited. None in, None out: a file without a
    space is not an error, it is a file that predates the section.
    """
    if not space:
        return None
    from copy import deepcopy

    from ConfigSpace import ConfigurationSpace

    decoded = ConfigurationSpace.from_serialized_dict(deepcopy(space))
    decoded.seed(seed)
    return decoded
