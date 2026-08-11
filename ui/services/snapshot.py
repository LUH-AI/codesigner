"""The single seam between the .ihpo snapshot format and the database.

`experiment_from_snapshot` builds (and saves) an Experiment row from a parsed
snapshot; `snapshot_from_experiment` produces a current-version snapshot dict
from a row. Import, export, and detail-page reconstruction all go through here.
"""

from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.core.files import File

from core import io
from core.provenance import (
    dataset_fingerprint, environment, evaluation, model_fingerprint,
)

from ..models import Experiment
from ..registry import OPTIMIZERS


def experiment_from_snapshot(snapshot: dict, dataset_file=None, model_file=None,
                             adopt_paths: bool = False, owner=None) -> Experiment:
    """Create and save an Experiment row from a parsed snapshot.

    Files are adopted into MEDIA from the uploads passed as *dataset_file* and
    *model_file*. Without them the row is created without that file — browsable,
    not runnable — which is what an imported `.ihpo` does until its dataset is
    attached.

    *adopt_paths* additionally allows `dataset_path` and `model_path` to be read
    from the snapshot as paths on this machine. It is off by default because a
    snapshot is only as trustworthy as wherever it came from: read from an
    uploaded file, those fields name any path the uploader likes, and adopting
    one copies a file they were never shown into an experiment they can export.
    Only a caller that produced the paths itself may turn it on — the create form
    (validated choices and a temp file it just wrote) and the import command
    (an operator naming a file on their own machine).

    A custom model .py is adopted only when ALLOW_CUSTOM_MODELS is on: untrusted
    code is never stored on an instance that has the feature disabled. Callers
    that pass *model_file* have already gated on the flag.

    *owner* is the account it belongs to, or None for an install with no
    accounts. Set at creation because there is nowhere else it could come
    from — an .ihpo has no notion of who made it.
    """
    exp = Experiment(
        name=snapshot["name"],
        model_name=snapshot["model_name"],
        optimizer_name=snapshot["optimizer_name"],
        optimizer_params=snapshot.get("optimizer_params", {}),
        metric_names=snapshot["metric_names"],
        primary_metric=snapshot.get("primary_metric"),
        original_metric=snapshot.get("original_metric"),
        seed=snapshot["seed"],
        cv_folds=int(snapshot.get("cv_folds") or 0),
        result=snapshot.get("result"),
        owner=owner,
    )

    if dataset_file is not None:
        name = getattr(dataset_file, "name", None) or _dataset_name(snapshot)
        exp.dataset.save(Path(name).name, dataset_file, save=False)
    elif adopt_paths:
        stored = snapshot.get("dataset_path", "")
        if stored and Path(stored).is_file():
            with open(stored, "rb") as fh:
                exp.dataset.save(Path(stored).name, File(fh), save=False)

    if model_file is not None:
        name = getattr(model_file, "name", None) or "model.py"
        exp.model_file.save(Path(name).name, model_file, save=False)
    elif adopt_paths and settings.ALLOW_CUSTOM_MODELS:
        stored_model = snapshot.get("model_path", "")
        if stored_model and Path(stored_model).is_file():
            with open(stored_model, "rb") as fh:
                exp.model_file.save(Path(stored_model).name, File(fh), save=False)

    exp.save()
    return exp


def snapshot_from_experiment(exp: Experiment, *, provenance: bool = False) -> dict:
    """Build a current-version .ihpo snapshot dict from an Experiment row.

    dataset_path points at the row's stored file, or is empty when it has none
    — a foreign path from an imported file is never echoed back out.

    *provenance* adds the sections that describe how the experiment was made
    rather than what it is: the data and model fingerprints, how a trial was
    evaluated, what the optimizer's settings resolved to, the history of runs,
    and the versions behind the numbers. Off by default because the run engine
    and the detail page rebuild through this function on every run and every
    page load, and the dataset fingerprint reads and hashes the file. Export
    turns it on; nothing else needs it.
    """
    snapshot = {
        "version": dist_version("codesigner"),
        "name": exp.name,
        "model_name": exp.model_name,
        "model_path": exp.model_file.path if exp.model_file else "",
        "optimizer_name": exp.optimizer_name,
        "optimizer_params": exp.optimizer_params,
        "primary_metric": exp.primary_metric,
        "original_metric": exp.original_metric,
        "metric_names": exp.metric_names,
        "seed": exp.seed,
        "cv_folds": exp.cv_folds,
        "dataset_path": exp.dataset.path if exp.dataset else "",
        "result": exp.result,
    }
    if provenance:
        snapshot.update(_provenance(exp))
    return snapshot


def _provenance(exp: Experiment) -> dict:
    """The sections that say how the experiment was made.

    Nested rather than flattened in beside the existing keys: they are a
    different kind of thing — a record of the process, not the configuration it
    ran under — and every one of them is optional on the way back in, so a file
    written before any of this existed still opens.
    """
    dataset = exp.dataset.path if exp.dataset else ""
    return {
        "data": dataset_fingerprint(dataset) if dataset else None,
        "model": model_fingerprint(
            exp.model_name, exp.model_file.path if exp.model_file else "", exp.env_meta),
        "evaluation": evaluation(exp.cv_folds, _target(dataset)),
        "optimizer": _optimizer_record(exp),
        "runs": [_run_record(index, run)
                 for index, run in enumerate(exp.runs.order_by("id"), start=1)],
        "environment": environment(),
    }


def _target(dataset_path: str):
    """The target column, for deciding whether the split could stratify."""
    if not dataset_path or not Path(dataset_path).is_file():
        return None
    try:
        _, y = io._load_frame(Path(dataset_path))
    except Exception:  # noqa: BLE001 — an unreadable dataset leaves it unanswered
        return None
    return y


def _optimizer_record(exp: Experiment) -> dict:
    """What the search was configured to do, with the blanks answered.

    `resolved` is the reader's copy: a blank setting means "whatever that
    component already does", which is the right thing to store and useless to
    read. Reconstruction still goes through `optimizer_params`, which is what
    was actually asked for — the difference matters if the installed SMAC ever
    changes a default, and `environment.packages.smac` says which one answered.

    The initial design is described here rather than sized here, because its
    size depends on the budget the run was given and that belongs to each run.
    """
    optimizer = OPTIMIZERS.get(exp.optimizer_name)
    params = exp.optimizer_params or {}
    resolved = None
    if optimizer is not None:
        kind = type(optimizer)
        try:
            resolved = kind(**kind.known_params(params)).resolved_params()
        except Exception:  # noqa: BLE001 — a record is never worth a failed export
            resolved = None
    return {
        "name": exp.optimizer_name,
        "resolved": resolved,
        "initial_design": {
            "kind": params.get("initial_design"),
            "sized_by": "trials" if params.get("exploration_trials") else "share",
            "trials": params.get("exploration_trials"),
            "share": params.get("exploration_ratio"),
        },
    }


def _run_record(index: int, run) -> dict:
    """One run: when, over which trials, under what, and why it ended."""
    offset = run.trial_offset
    count = run.trial_count
    span = None
    if offset is not None and count:
        span = [offset + 1, offset + count]
    return {
        "index": index,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "trial_range": span,
        "primary_metric": run.primary_metric,
        "optimizer_params": run.optimizer_params or None,
        "stopping": run.stopping,
        "stopped_by": run.stopped_by or None,
        # The budget SMAC was told about, which is what sized its initial
        # design. Not the same as the trial cap once a run resumes.
        "budget_told": ((offset or 0) + run.stopping["max_trials"]
                        if run.stopping.get("max_trials") else None),
        "events": run.events or [],
    }


def _dataset_name(snapshot: dict) -> str:
    stored = snapshot.get("dataset_path", "")
    return Path(stored).name if stored else "dataset.csv"
