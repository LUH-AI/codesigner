"""JSON-based experiment serialization and dataset/model loading helpers.

File format (.ihpo): UTF-8 JSON with the following top-level keys:
    version          str   — application version string (e.g. "0.1.0")
    name             str
    model_name       str   — key in the MODELS registry (or the model's .name
                             property when a custom model file is used)
    model_path       str   — path to a custom model .py file, or "" for
                             registry models
    optimizer_name   str   — optimizer.name (key in OPTIMIZERS registry)
    optimizer_params dict  — {param_name: value} for each params_schema entry
    primary_metric   str | null
    original_metric  str | null
    metric_names     list[str]
    seed             int
    dataset_path     str   — path to the source CSV
    result           dict | null — serialized OptimizationResult
                             (runhistory-mirrored, see BaseOptimizer.serialize_result)

The dataset arrays and model object are NOT stored.  On load, dataset_path and
model_path are resolved from disk; missing files must be re-supplied by the
caller (or the experiment loaded read-only).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .version import VERSION as _VERSION

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


def _load_splits(csv_path: Path, seed: int):
    """Reconstruct the identical train/val split from a CSV path + seed."""
    raw = csv_path.read_bytes()
    sample = raw[:2048].decode("utf-8", errors="replace")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(csv_path, sep=sep)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    try:
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=0.2, random_state=seed)


# ── Public API ────────────────────────────────────────────────────────────────

def save(name: str, exp: dict) -> bytes:
    """Serialize *exp* to UTF-8 JSON bytes."""
    snapshot = {
        "version":          _VERSION,
        "name":             name,
        "model_name":       exp["model_name"],
        "model_path":       exp.get("model_path", ""),
        "optimizer_name":   exp["optimizer"].name,
        "optimizer_params": exp["optimizer"].get_params(),
        "primary_metric":   exp["primary_metric"],
        "original_metric":  exp["original_metric"],
        "metric_names":     list(exp["metrics"].keys()),
        "seed":             exp["seed"],
        "dataset_path":     exp.get("dataset_path", ""),
        "result":           exp["optimizer"].serialize_result(exp["result"]) if exp["result"] is not None else None,
    }
    return json.dumps(snapshot, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")


def parse(data: bytes) -> dict:
    """Decode and validate raw bytes; return the snapshot dict.

    Raises ``ValueError`` with a human-readable message on any problem.
    Does NOT reconstruct any objects — call :func:`build_experiment` for that.
    """
    try:
        snapshot = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc

    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("version"), str):
        raise ValueError("not a valid .ihpo file or unsupported version")

    for key in ("name", "model_name", "optimizer_name", "optimizer_params",
                "metric_names", "seed", "dataset_path"):
        if key not in snapshot:
            raise ValueError(f"missing field: {key!r}")

    return snapshot


def dataset_path_ok(snapshot: dict) -> bool:
    """Return True if the stored dataset_path exists and is a readable file."""
    p = snapshot.get("dataset_path", "")
    return bool(p) and Path(p).is_file()


def model_path_ok(snapshot: dict) -> bool:
    """Return True if the model is a registry model (no path) or its file exists."""
    p = snapshot.get("model_path", "")
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

    Returns ``(name, exp_dict)`` on success.  Raises ``ValueError`` on failure.
    """
    model_path = snapshot.get("model_path", "")

    missing_metrics = [m for m in snapshot["metric_names"] if m not in available_metrics]
    if missing_metrics:
        raise ValueError(f"unknown metric(s): {', '.join(missing_metrics)}")

    opt_name = snapshot["optimizer_name"]
    opt_entry = next(
        (o for o in available_optimizers.values() if o.name == opt_name),
        None,
    )
    if opt_entry is None:
        raise ValueError(f"optimizer '{opt_name}' is not available")

    seed = snapshot["seed"]

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_path:
        if read_only:
            model = None
        else:
            model, err = load_model_from_path(model_path)
            if err:
                raise ValueError(f"custom model error: {err}")
        resolved_model = snapshot["model_name"]
    else:
        stored_model = model_name if model_name is not None else snapshot["model_name"]
        model_entry = (
            available_models.get(stored_model)
            or next((m for m in available_models.values() if m.name == stored_model), None)
        )
        if model_entry is None:
            raise ValueError(f"model '{stored_model}' is not available")
        model = model_entry
        resolved_model = stored_model

    # ── Dataset ───────────────────────────────────────────────────────────────
    if read_only:
        X_train = X_val = y_train = y_val = None
    else:
        path = Path(snapshot["dataset_path"])
        if not path.is_file():
            raise ValueError(f"dataset not found: {snapshot['dataset_path']}")
        X_train, X_val, y_train, y_val = _load_splits(path, seed)

    metrics   = {m: available_metrics[m] for m in snapshot["metric_names"]}
    opt_type  = type(opt_entry)
    optimizer = opt_type(**snapshot.get("optimizer_params", {}))
    result    = optimizer.deserialize_result(snapshot["result"]) if snapshot.get("result") else None

    exp = {
        "model":           model,
        "model_name":      resolved_model,
        "model_path":      model_path,
        "optimizer":       optimizer,
        "primary_metric":  snapshot.get("primary_metric"),
        "original_metric": snapshot.get("original_metric"),
        "metrics":         metrics,
        "seed":            seed,
        "dataset_path":    snapshot["dataset_path"],
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "result":  result,
    }
    return snapshot["name"], exp
