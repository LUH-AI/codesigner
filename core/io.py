"""`core.io` reads and writes the `.ihpo` file, UTF-8 JSON with one object per
subject (format, version, name, seed, dataset, model, evaluation, metrics,
optimizer, optional space and priors, runs, environment) and `result` last as
the longest. `result` is a strict superset of SMAC's runhistory.json, and
`core.optimizers.smac.importer` fills the other fields from a SMAC directory.
The file only keeps the dataset and model paths, and a caller must resupply a
missing file or load the experiment read-only. Every reader goes through
`normalize`, which lifts older flat files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import encoding, tasks
from .datasets import DEFAULT_TEST_SIZE, Evaluation, load_evaluation, load_splits
from .jsonable import finite
from .metrics import declarations
from .model_loading import give_context, resolve_model
from .models.base import Tunable
from .paths import MAX_STATE_FILES, is_safe_relative
from .processing import choices_of
from .registry import optimizer_class
from .splits import MIN_FOLDS


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


#: Format 1 was one flat namespace that named some settings twice, and format 2
#: states each subject once. A file without the key is format 1.
SNAPSHOT_FORMAT = 2


def _check_format(snapshot: dict) -> None:
    """`normalize` would push a future file through the format 1 lifter, which
    returns mostly empty sections and fails screens later.
    """
    written_as = snapshot.get("format")
    if isinstance(written_as, int) and written_as > SNAPSHOT_FORMAT:
        raise ValueError(
            f"this file is in format {written_as}, which is newer than the "
            f"{SNAPSHOT_FORMAT} this version of codesigner understands")


def normalize(snapshot: dict) -> dict:
    """Every snapshot reader calls `normalize`, so the application knows one
    shape. A format 1 file is lifted, minus its record-only `optimizer.resolved`
    and per-run settings copies, since an export rebuilds the record.
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
    """`_lifted_evaluation` derives `scheme`, `folds` and `test_size` from
    `cv_folds` and keeps the recorded `stratified` value.
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
    """`evaluation_of` fills every setting: `folds` 0 means a holdout,
    `test_size` keeps `DEFAULT_TEST_SIZE` under cross-validation for a later
    scheme switch, and a positive `horizon` makes a forecast."""
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
    """`save` and the export view both go through `to_bytes`, so neither can
    write a non-finite number.
    """
    return json.dumps(finite(snapshot), ensure_ascii=False, indent=2,
                      default=_json_default).encode("utf-8")


def parse(data: bytes) -> dict:
    """`parse` raises ``ValueError`` with a readable message; `build_experiment`
    reconstructs the objects.
    """
    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("version"), str):
        raise ValueError("not a valid .ihpo file or unsupported version")

    # A newer file would otherwise report a missing format 1 field instead of
    # the real problem.
    _check_format(raw)

    # A missing field is reported under the name the file itself uses.
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
    """Deserializing writes each `optimizer_state` file to its relative path,
    so `check_optimizer_state` raises a `ValueError` for an edited path at read
    time.
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
    """A trial missing a declared metric would raise a KeyError at render time,
    a 500 on the detail page. Like `BaseOptimizer.deserialize_result`,
    `_check_trial_scores` reads an entry without `scores` as the primary metric
    alone.
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
    p = (normalize(snapshot).get("dataset") or {}).get("path", "")
    return bool(p) and Path(p).is_file()


def model_path_ok(snapshot: dict) -> bool:
    p = (normalize(snapshot).get("model") or {}).get("path", "")
    return not p or Path(p).is_file()


def attach_dataset(exp: dict, dataset_path: str) -> None:
    """`attach_dataset` raises ``ValueError`` when the file cannot be read.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"dataset not found: {dataset_path}")
    X_train, X_val, y_train, y_val = load_splits(
        path, exp["seed"], exp.get("test_size") or DEFAULT_TEST_SIZE,
        exp.get("task", tasks.DEFAULT), exp.get("target_column") or None)
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
    """*model_name* overrides the stored built-in name and is ignored for a file
    model. *read_only* skips all file loading, and *load_model* False only skips
    the custom model import. The result is ``(name, exp)``, and a failure raises
    ``ValueError``.
    """
    snapshot = normalize(snapshot)
    ev = evaluation_of(snapshot)
    metric_names = snapshot["metrics"]["names"]
    model_path = snapshot["model"].get("path", "")
    task = tasks.task_of(snapshot)
    # A file that names no target column predicts its dataset's last column.
    target = str((snapshot.get("dataset") or {}).get("target_column") or "") or None

    # A file may declare its own metrics, so an imported run with an unknown
    # objective still opens; `.undescribe` gives them a scorer that raises.
    declared = declarations((snapshot.get("result") or {}).get("declared_metrics"))
    missing_metrics = [m for m in metric_names
                       if m not in available_metrics and m not in declared]
    if missing_metrics:
        raise ValueError(f"unknown metric(s): {', '.join(missing_metrics)}")

    opt_name = snapshot["optimizer"]["name"]
    # Aliases keep renamed optimizers loadable from old files.
    opt_type = optimizer_class(opt_name, available_optimizers)
    if opt_type is None:
        raise ValueError(f"optimizer '{opt_name}' is not available")

    seed = snapshot["seed"]

    model, resolved_model = resolve_model(snapshot, ev, task, available_models, model_name,
                                          load_file=not read_only and load_model)

    # `splits` defines what a trial is evaluated over; the four arrays stay for
    # their many callers.
    data = Evaluation()
    if not read_only:
        dataset_path = snapshot["dataset"].get("path", "")
        path = Path(dataset_path)
        if not path.is_file():
            raise ValueError(f"dataset not found: {dataset_path}")
        data = load_evaluation(path, ev, seed, task, target)
        if model is not None:
            give_context(model, "feature_kinds", tuple(encoding.feature_kinds(data.columns)))
            # Each built-in model gets the processing plan, while an uploaded
            # model gets its columns unprocessed.
            processing = choices_of((snapshot.get("dataset") or {}).get("processing"))
            for told in (getattr(model, "model", None), model):
                if isinstance(told, Tunable):
                    told.processing = dict(processing)
            if data.forecast is not None:
                give_context(model, "forecast", dict(data.forecast))

    # The registry wins, since a later build may correct a metric.
    metrics   = {m: available_metrics.get(m) or declared[m] for m in metric_names}
    # A withdrawn or renamed parameter would otherwise raise as an unexpected
    # keyword.
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
        "target_column":   target or "",
        "X_train": data.X_train, "y_train": data.y_train,
        "X_val":   data.X_val,   "y_val":   data.y_val,
        "cv_folds": ev["folds"],
        "test_size": ev["test_size"],
        "time_column": ev["time_column"],
        "gap": ev["gap"],
        "task":    task,
        "splits":  data.splits,
        # `core.encoding` spec; None when nothing was read.
        "encoding": data.columns,
        "feature_kinds": encoding.feature_kinds(data.columns) if data.columns else None,
        # None outside a forecast.
        "forecast": data.forecast,
        # Each split row's time, forecasts only.
        "times": data.times,
        # The serialized dict, since `config_space_from_serialized` decodes and
        # seeds it.
        "config_space": snapshot.get("space"),
        "result":  result,
    }
    return snapshot["name"], exp


def config_space_from_serialized(space: dict | None, seed: int = 0):
    """`from_serialized_dict` pops each hyperparameter's `type`, so
    `config_space_from_serialized` decodes a copy of the caller's dict and
    reapplies the lost seed. `config_space_from_serialized` gives None for a
    missing space, since older files predate the section.
    """
    if not space:
        return None
    from copy import deepcopy

    from ConfigSpace import ConfigurationSpace

    decoded = ConfigurationSpace.from_serialized_dict(deepcopy(space))
    decoded.seed(seed)
    return decoded
