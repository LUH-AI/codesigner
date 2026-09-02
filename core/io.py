"""JSON-based experiment serialization and dataset/model loading helpers.

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
    optimizer   dict  — name, params, and (on export) defaults_used
    space       dict | absent — the search space, as ConfigSpace's own
                        serialized dict; byte-identical to SMAC's
                        configspace.json. Optional: a file without one falls
                        back to asking the model, which is what every file
                        written before this section existed does.
    runs        list  — export only: the history of runs over the trials
    environment dict  — export only: the versions behind the numbers
    result      dict | null — serialized OptimizationResult, a strict superset
                        of SMAC's runhistory.json (see serialize_result)

The dataset arrays and model object are NOT stored. On load the two paths are
resolved from disk; missing files must be re-supplied by the caller (or the
experiment loaded read-only).

Every reader goes through :func:`normalize`, so an older file — which spelled
all of this as one flat namespace — is lifted rather than refused.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import tempfile
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from . import provenance
from .metrics import declarations
from .paths import MAX_STATE_FILES, is_safe_relative
from .splits import MIN_FOLDS, cross_validation, holdout

_REPO_ROOT   = Path(__file__).parent.parent
_DATASETS_DIR = _REPO_ROOT / "datasets"
_MODELS_DIR   = _REPO_ROOT / "mounted_models"


# ── Dataset / model discovery ─────────────────────────────────────────────────

def demo_datasets() -> dict[str, str]:
    """Return {stem: path} for CSVs in datasets/.

    Files mounted to datasets/ at runtime (e.g. a Docker volume) appear in the
    dropdown automatically alongside the bundled demo CSVs.
    """
    result: dict[str, str] = {}
    if _DATASETS_DIR.exists():
        for p in sorted(_DATASETS_DIR.glob("*.csv")):
            result[p.stem] = str(p)
    return result


def mounted_models() -> dict[str, str]:
    """Return {stem: path} for .py files in mounted_models/.

    Files mounted to mounted_models/ at runtime appear in the dropdown
    automatically.
    """
    result: dict[str, str] = {}
    if _MODELS_DIR.exists():
        for p in sorted(_MODELS_DIR.glob("*.py")):
            result[p.stem] = str(p)
    return result


# ── Custom model loading ──────────────────────────────────────────────────────

def load_model_from_path(model_path: str) -> tuple:
    """Load a BaseModel subclass from a .py file at *model_path*.

    Executes the file as Python — callers are responsible for only passing
    trusted paths (see the ALLOW_CUSTOM_MODELS setting).

    Returns ``(instance, None)`` on success or ``(None, error_str)`` on failure.
    Error strings are plain English and not translated.
    """
    from .models import BaseModel

    path = Path(model_path)
    if not path.is_file():
        return None, f"File not found: {model_path}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as f:
        f.write(path.read_bytes())
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_custom_model", tmp_path)
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
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_csv(csv_path: Path):
    """The dataset as a DataFrame, with the separator sniffed."""
    raw = csv_path.read_bytes()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(csv_path, sep=sep)


def _load_frame(csv_path: Path):
    """The dataset as (X, y). The last column is the target."""
    df = _read_csv(csv_path)
    return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()


#: The share of a dataset held out for validation when the scheme is a single
#: split, if the experiment does not say. Every experiment written before the
#: share was settable used this, so it is also what those files mean.
DEFAULT_TEST_SIZE = 0.2


def _load_splits(csv_path: Path, seed: int, test_size: float = DEFAULT_TEST_SIZE):
    """Reconstruct the identical train/val split from a CSV path, seed and
    validation share."""
    X, y = _load_frame(csv_path)
    size = float(test_size or DEFAULT_TEST_SIZE)
    try:
        return train_test_split(X, y, test_size=size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=size, random_state=seed)


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
    for section in ("space", "runs", "environment"):
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


def folds_of(snapshot: dict) -> int:
    """How many folds this snapshot asks for; 0 is a single holdout."""
    return int((snapshot.get("evaluation") or {}).get("folds") or 0)


def test_size_of(snapshot: dict) -> float:
    """What share of the data this snapshot holds out, when it holds out one.

    Meaningless under cross-validation, where every row is validated against
    once; `DEFAULT_TEST_SIZE` there rather than None so the column always holds
    a number and a later switch of scheme has something to start from.
    """
    section = snapshot.get("evaluation") or {}
    return float(section.get("test_size") or DEFAULT_TEST_SIZE)


def finite(value):
    """*value* with every non-finite number in it replaced by None, recursively.

    `json.dumps` writes `inf` and `nan` as the bare tokens `Infinity` and `NaN`,
    which Python reads back and a strict parser refuses — so an `.ihpo`
    containing one is not the portable file it claims to be. Two sources put
    them there and neither is a measurement:

    - SMAC's `scenario.json`, carried verbatim in `optimizer_state`, whose
      `crash_cost` and `walltime_limit` default to infinity. Every `.ihpo` ever
      written for a SMAC run carries four of them.
    - a run imported from somebody else's output, where SMAC writes the same
      tokens itself.

    None rather than a large number: infinity is not a value here, it is the
    absence of a limit, and nothing reads these fields back. `default=` on
    `json.dumps` cannot do this — it is never called for a float.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def to_bytes(snapshot: dict) -> bytes:
    """*snapshot* as the bytes of an `.ihpo` file.

    The one place a snapshot becomes a file, so `save` and the export view
    cannot disagree about what one may contain — see `finite`.
    """
    return json.dumps(finite(snapshot), ensure_ascii=False, indent=2,
                      default=_json_default).encode("utf-8")


def save(name: str, exp: dict) -> bytes:
    """Serialize *exp* to UTF-8 JSON bytes."""
    model_path = exp.get("model_path", "")
    snapshot = {
        "format":     SNAPSHOT_FORMAT,
        "version":    _dist_version("codesigner"),
        "name":       name,
        "seed":       exp["seed"],
        "dataset":    {"filename": Path(exp.get("dataset_path", "")).name,
                       "path": exp.get("dataset_path", "")},
        "model":      {"kind": "file" if model_path else "registry",
                       "name": exp["model_name"], "path": model_path},
        # Shared with ui/services/snapshot.py's live export path, so the
        # kfold/holdout decision has exactly one place to look, not two that
        # can quietly disagree.
        "evaluation": provenance.evaluation(exp.get("cv_folds") or 0,
                                            test_size=exp.get("test_size")),
        "metrics":    {"names": list(exp["metrics"].keys()),
                       "current": exp["current_metric"],
                       "original": exp["original_metric"]},
        "optimizer":  {"name": exp["optimizer"].name,
                       "params": exp["optimizer"].get_params()},
        "result":     exp["optimizer"].serialize_result(exp["result"]) if exp["result"] is not None else None,
    }
    if exp.get("config_space"):
        snapshot["space"] = exp["config_space"]
    return to_bytes(snapshot)


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
    _check_optimizer_state(snapshot)
    _check_trial_scores(snapshot)
    return snapshot


def _check_optimizer_state(snapshot: dict) -> None:
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
    X_train, X_val, y_train, y_val = _load_splits(path, exp["seed"])
    exp["X_train"]      = X_train
    exp["y_train"]      = y_train
    exp["X_val"]        = X_val
    exp["y_val"]        = y_val
    exp["dataset_path"] = str(path.resolve())


def attach_model(exp: dict, model_path: str) -> None:
    """Load a custom model from *model_path* and update *exp* in-place.

    Raises ``ValueError`` if the file cannot be found or is not a valid model.
    """
    model, err = load_model_from_path(model_path)
    if err:
        raise ValueError(err)
    exp["model"]      = model
    exp["model_name"] = model.name
    exp["model_path"] = str(Path(model_path).resolve())


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
    metric_names = snapshot["metrics"]["names"]
    model_path = snapshot["model"].get("path", "")

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
    opt_entry = next(
        (o for o in available_optimizers.values()
         if o.name == opt_name or opt_name in getattr(o, "aliases", ())),
        None,
    )
    if opt_entry is None:
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
        resolved_model = snapshot["model"]["name"]
    else:
        stored_model = model_name if model_name is not None else snapshot["model"]["name"]
        model_entry = (
            available_models.get(stored_model)
            or next((m for m in available_models.values() if m.name == stored_model), None)
        )
        if model_entry is None:
            raise ValueError(f"model '{stored_model}' is not available")
        model = model_entry
        resolved_model = stored_model

    # ── Dataset ───────────────────────────────────────────────────────────────
    # `splits` is what a trial is actually evaluated over — one fold for a
    # holdout, k for cross-validation. The four arrays stay as they were: plenty
    # of callers read them, and for a holdout they are the same data.
    cv_folds = folds_of(snapshot)
    test_size = test_size_of(snapshot)
    if read_only:
        X_train = X_val = y_train = y_val = None
        splits = None
    else:
        dataset_path = snapshot["dataset"].get("path", "")
        path = Path(dataset_path)
        if not path.is_file():
            raise ValueError(f"dataset not found: {dataset_path}")
        X_train, X_val, y_train, y_val = _load_splits(path, seed, test_size)
        if cv_folds >= MIN_FOLDS:
            X_all, y_all = _load_frame(path)
            splits = cross_validation(X_all, y_all, cv_folds, seed)
        else:
            splits = holdout(X_train, y_train, X_val, y_val)

    # The registry first: a later build correcting what a metric means must not
    # be overruled by a file written before the correction.
    metrics   = {m: available_metrics.get(m) or declared[m] for m in metric_names}
    opt_type  = type(opt_entry)
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
        "splits":  splits,
        # Undecoded. Kept as the serialized dict the file carries, because the
        # one caller that wants a live object also wants it seeded, and
        # `config_space_from_serialized` does both.
        "config_space": snapshot.get("space"),
        "result":  result,
    }
    return snapshot["name"], exp


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
