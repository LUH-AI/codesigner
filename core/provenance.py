"""What an `.ihpo` records about how its experiment was made.

An `.ihpo` is meant to be a superset of SMAC's own output — and for SMAC's own
files it already was: `result` mirrors the runhistory and `result.optimizer_state`
embeds `intensifier.json`, `optimization.json`, `scenario.json` and
`configspace.json` verbatim. What was missing is everything *around* the
optimizer: which data, which model, how a trial was evaluated, what the search
was actually configured to do, and what happened over the runs that got the
experiment to where it is.

This module computes those sections. They are additive — every key that existed
before is untouched and in the same place, so a file written today opens in an
older build and a file written by an older build opens here.

**Fingerprints, not contents.** The dataset and the model are recorded by
SHA-256, not embedded. The file stays a record rather than an archive, and the
digest does work: attaching a dataset that disagrees with it is refused, so an
experiment cannot quietly be resumed against different data and have the new
trials compared with the old. The cost is that a custom-model `.ihpo` needs its
`.py` supplied alongside.

**One seed.** `seed` stays a single top-level field because that is what it is:
`Experiment.seed` drives the split, the config-space sampler, SMAC's scenario
(which fans it out to the surrogate, maximizer, initial design, random design
and intensifier), `fit_predict` and the importance computation. One source of
randomness for the whole experiment is what makes it reproducible from
(dataset, seed, setup), and splitting it across sections would suggest
otherwise.
"""

from __future__ import annotations

import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from typing import Any, Dict, Optional

from .splits import MIN_FOLDS

#: Read in blocks rather than whole: a dataset is not always small, and the
#: digest is computed on every export.
_DIGEST_BLOCK = 1 << 20

#: The libraries a result depends on for its numbers. Recorded so a file that
#: reproduces differently later can be told apart from one that reproduces
#: differently *here* — a surrogate's defaults are its version's defaults, and
#: an unset setting means whichever ones these had.
_TRACKED_PACKAGES = ("smac", "scikit-learn", "ConfigSpace", "numpy", "pandas",
                     "scipy")


def _version(package: str) -> Optional[str]:
    try:
        return _dist_version(package)
    except PackageNotFoundError:
        return None


def sha256(path) -> str:
    """The digest of a file, or "" when there is no file to digest."""
    path = Path(path)
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        return sha256_stream(fh)


def sha256_stream(fh) -> str:
    """The digest of an open file, read from the start and rewound after.

    An upload arrives as a stream rather than a path, and whoever handed it over
    still has to be able to save it — so this leaves the position where it found
    it rather than at the end.
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


def dataset_fingerprint(path, frame=None) -> Dict[str, Any]:
    """What the dataset was, without the dataset.

    Enough to recognise it — the digest — and enough to read the record without
    it: how many rows and columns, what they were called, and which one was the
    target. *frame* is the already-loaded (X, y) if the caller has it, so an
    export does not read the file twice.
    """
    path = Path(path)
    record: Dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256(path),
        "rows": None, "columns": None,
        "column_names": None, "target_column": None,
    }
    if not path.is_file():
        return record

    import pandas as pd

    from .io import _read_csv

    try:
        df = _read_csv(path) if frame is None else frame
    except Exception:  # noqa: BLE001 — an unreadable file still gets its digest
        return record
    if not isinstance(df, pd.DataFrame) or df.empty:
        return record

    names = [str(c) for c in df.columns]
    record.update({
        "rows": int(len(df)),
        "columns": int(len(names)),
        "column_names": names,
        # The last column is the target, everywhere in this application.
        "target_column": names[-1] if names else None,
    })
    return record


def model_fingerprint(model_name: str, model_path: str, env_meta=None) -> Dict[str, Any]:
    """Which model, and what it needed to run.

    A registry model is named and nothing else — it ships with the application,
    so its version is the application's. An uploaded one is a file, and what
    matters about it is its digest and the environment that was locked for it.
    """
    env_meta = env_meta or {}
    if not model_path:
        return {"kind": "registry", "name": model_name, "sha256": None,
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


def evaluation(cv_folds: int, y=None) -> Dict[str, Any]:
    """How a trial was evaluated — fixed for the experiment's life.

    `stratified` is *resolved*, not intended. Both schemes ask for stratification
    and fall back to a plain division when scikit-learn refuses the target,
    which in practice means a continuous one; recording the request rather than
    the outcome would put a claim in the file that the run did not honour. It is
    left null when the target was not to hand.
    """
    folds = int(cv_folds or 0)
    return {
        "scheme": "kfold" if folds >= MIN_FOLDS else "holdout",
        "folds": folds if folds >= MIN_FOLDS else None,
        "test_size": None if folds >= MIN_FOLDS else 0.2,
        "stratified": None if y is None else _stratifiable(y),
    }


def _stratifiable(y) -> bool:
    """Whether scikit-learn will stratify on this target."""
    try:
        from sklearn.model_selection import train_test_split

        train_test_split(range(len(y)), test_size=0.2, random_state=0, stratify=y)
        return True
    except Exception:  # noqa: BLE001 — refusing is the answer, not an error
        return False


def environment() -> Dict[str, Any]:
    """What produced the numbers in this file."""
    return {
        "codesigner": _version("codesigner"),
        "python": sys.version.split()[0],
        "packages": {name: _version(name) for name in _TRACKED_PACKAGES},
    }


def dataset_mismatch(recorded, actual: str) -> str:
    """Why *actual* is not the digest this file records, or "" if it is.

    Refusing is the point of recording the digest. An experiment resumed against
    different data would go on adding trials to a history they cannot be
    compared with, and nothing downstream — not the incumbent, not the
    surrogate, not the importance — has any way to notice: every trial would
    look exactly as valid as the ones before it.

    Two things are deliberately *not* refusals. A file with no fingerprint —
    anything exported before this existed — has nothing to disagree with. And a
    file imported without a dataset at all is browsable and not runnable, which
    is the case this has always supported; the check happens when one is
    attached, which is the moment it could start to matter.
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
