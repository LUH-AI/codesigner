"""`provenance` computes the `.ihpo` sections around the optimizer: data, model,
evaluation, search setup and run history. The sections are additive, so older
and newer builds open each other's files.

The dataset and model are recorded by SHA-256 fingerprint. Therefore a
custom-model `.ihpo` needs its `.py` supplied alongside. `seed` stays one
top-level field because one seed drives the split, the sampler, the SMAC
scenario, `fit_predict` and the importance computation.
"""

from __future__ import annotations

import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from typing import Any, Dict, Optional

from .splits import MIN_FOLDS
from .tasks import CLASSIFICATION

#: Blocks, since exports digest datasets of any size.
_DIGEST_BLOCK = 1 << 20

#: Library versions explain a result that later reproduces differently, since
#: surrogate defaults follow the version.
_TRACKED_PACKAGES = ("smac", "scikit-learn", "ConfigSpace", "numpy", "pandas",
                     "scipy")


def _version(package: str) -> Optional[str]:
    try:
        return _dist_version(package)
    except PackageNotFoundError:
        return None


def sha256(path) -> str:
    """`sha256` returns an empty string for a missing file."""
    path = Path(path)
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        return sha256_stream(fh)


def sha256_stream(fh) -> str:
    """`sha256_stream` restores the stream position so the uploader can still
    save the upload.
    """
    where = fh.tell() if hasattr(fh, "tell") else None
    if hasattr(fh, "seek"):
        fh.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: fh.read(_DIGEST_BLOCK), b""):
        digest.update(block)
    if hasattr(fh, "seek"):
        fh.seek(where or 0)
    return digest.hexdigest()


def dataset_fingerprint(path, frame=None, target: str = "") -> Dict[str, Any]:
    """`dataset_fingerprint` records the digest, shape, column names and target,
    so the record reads without the file. *frame* spares an export a second
    read, and *target* names the column predicted.
    """
    path = Path(path)
    record: Dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256(path),
        "rows": None, "columns": None,
        "column_names": None, "target_column": None,
        "column_kinds": None,
    }
    if not path.is_file():
        return record

    import pandas as pd

    from .columns import infer_kinds
    from .datasets import read_dataset

    try:
        df = read_dataset(path, target or None) if frame is None else frame
    except Exception:  # noqa: BLE001 — an unreadable file still gets its digest
        return record
    if not isinstance(df, pd.DataFrame) or df.empty:
        return record

    names = [str(c) for c in df.columns]
    record.update({
        "rows": int(len(df)),
        "columns": int(len(names)),
        "column_names": names,
        # The target is read last, as everywhere in the app.
        "target_column": names[-1] if names else None,
        # Each feature's kind decides what the model received (`core.encoding`).
        "column_kinds": infer_kinds(df.iloc[:, :-1]),
    })
    return record


def model_fingerprint(model_name: str, model_path: str, env_meta=None,
                      kind: str = "") -> Dict[str, Any]:
    """A registry model only records its name, and an uploaded one its digest
    and locked environment. The caller passes *kind* for an imported run, which
    `model_path` alone would mislabel as a registry model.
    """
    env_meta = env_meta or {}
    if not model_path:
        return {"kind": kind or "registry", "name": model_name, "sha256": None,
                "dependencies": None, "requires_python": None,
                "python": None, "lock_sha256": None}
    return {
        "kind": "file",
        "name": model_name,
        "sha256": sha256(model_path),
        "dependencies": env_meta.get("dependencies"),
        "requires_python": env_meta.get("requires_python"),
        "python": env_meta.get("python"),
        "lock_sha256": env_meta.get("lock_sha256"),
    }


def evaluation(cv_folds: int, y=None, test_size: Optional[float] = None,
               task: str = CLASSIFICATION, time_column: str = "",
               gap: int = 0, horizon: int = 0, series_column: str = "",
               season: int = 0) -> Dict[str, Any]:
    """`evaluation` records how trials were scored, fixed for the experiment's
    life. *time_column* and *gap* describe a time-ordered split, and a forecast
    adds *horizon*, *series_column* and *season* with one backtest per fold.

    `stratified` records the outcome, because both schemes fall back to a plain
    split when scikit-learn raises a `ValueError` for the target. The field is
    null when the target was unavailable and always False for time-ordered or
    regression splits.
    """
    folds = int(cv_folds or 0)
    forecast = bool(horizon) and bool(time_column)
    if forecast:
        # One backtest still counts as a backtest.
        folds = max(folds, 1)
    return {
        "scheme": "backtest" if forecast else "kfold" if folds >= MIN_FOLDS else "holdout",
        "folds": folds if forecast or folds >= MIN_FOLDS else None,
        # Null where nothing is held out by share.
        "test_size": (None if forecast or folds >= MIN_FOLDS
                      else float(test_size or 0.2)),
        "stratified": (False if task != CLASSIFICATION or time_column
                       else None if y is None else _stratifiable(y)),
        "task": task,
        "time_column": time_column or None,
        "gap": int(gap or 0) if time_column else None,
        # Null except for a forecast.
        "horizon": int(horizon) if forecast else None,
        "series_column": (series_column or None) if forecast else None,
        "season": (int(season or 0) or None) if forecast else None,
    }


def _stratifiable(y) -> bool:
    try:
        from sklearn.model_selection import train_test_split

        train_test_split(range(len(y)), test_size=0.2, random_state=0, stratify=y)
        return True
    except Exception:  # noqa: BLE001 — an exception means no stratified split exists
        return False


def environment() -> Dict[str, Any]:
    return {
        "codesigner": _version("codesigner"),
        "python": sys.version.split()[0],
        "packages": {name: _version(name) for name in _TRACKED_PACKAGES},
    }


def dataset_mismatch(recorded, actual: str) -> str:
    """`dataset_mismatch` reports a mismatch because trials on different data
    would join an incomparable history and nothing downstream would notice. A
    file without a fingerprint, or imported without a dataset, passes; the check
    runs when a dataset is attached.
    """
    expected = (recorded or {}).get("sha256")
    if not expected or not actual or actual == expected:
        return ""
    name = (recorded or {}).get("filename") or "the dataset"
    return (f"This is not the dataset the experiment was run on: {name} was "
            f"recorded as {expected[:12]}… and this file is {actual[:12]}…. "
            f"Trials measured on different data cannot be compared with each "
            f"other, so the experiment would go on adding to a history it does "
            f"not belong to.")
