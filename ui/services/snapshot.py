"""The single seam between the .ihpo snapshot format and the database.

`experiment_from_snapshot` builds (and saves) an Experiment row from a parsed
snapshot; `snapshot_from_experiment` produces a current-version snapshot dict
from a row. Import, export, and detail-page reconstruction all go through here.
"""

from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.utils.dateparse import parse_datetime

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

    *adopt_paths* additionally allows the snapshot's `dataset.path` and
    `model.path` to be read as paths on this machine. It is off by default because a
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
    snapshot = io.normalize(snapshot)
    exp = Experiment(
        name=snapshot["name"],
        model_name=snapshot["model"]["name"],
        optimizer_name=snapshot["optimizer"]["name"],
        optimizer_params=snapshot["optimizer"].get("params") or {},
        metric_names=snapshot["metrics"]["names"],
        current_metric=snapshot["metrics"].get("current"),
        original_metric=snapshot["metrics"].get("original"),
        seed=snapshot["seed"],
        cv_folds=io.folds_of(snapshot),
        result=snapshot.get("result"),
        owner=owner,
    )

    if dataset_file is not None:
        name = getattr(dataset_file, "name", None) or _dataset_name(snapshot)
        exp.dataset.save(Path(name).name, dataset_file, save=False)
    elif adopt_paths:
        stored = snapshot["dataset"].get("path", "")
        if stored and Path(stored).is_file():
            with open(stored, "rb") as fh:
                exp.dataset.save(Path(stored).name, File(fh), save=False)

    if model_file is not None:
        name = getattr(model_file, "name", None) or "model.py"
        exp.model_file.save(Path(name).name, model_file, save=False)
    elif adopt_paths and settings.ALLOW_CUSTOM_MODELS:
        stored_model = snapshot["model"].get("path", "")
        if stored_model and Path(stored_model).is_file():
            with open(stored_model, "rb") as fh:
                exp.model_file.save(Path(stored_model).name, File(fh), save=False)

    exp.save()
    _restore_runs(exp, snapshot.get("runs") or [])
    return exp


#: A run that had not finished when the file was written did not finish at all —
#: it was interrupted by whatever ended the process that was running it. Imported
#: as pending or running it would leave `Experiment.is_running` true for ever and
#: the detail page polling a run that cannot report.
_UNFINISHED = ("pending", "running")


def _restore_runs(exp: Experiment, recorded: list) -> None:
    """Rebuild the experiment's run history from what the file recorded.

    Without this the `runs` section is write-only: an imported experiment keeps
    its trials and loses which run produced which of them, what bounded each
    one, why it ended and when the metric changed — all of which are in the
    file. A round trip has to be lossless or the file is not the record it
    claims to be.

    `started_by` is deliberately not restored. An account on the instance that
    exported this is not an account here, and inventing a local one would put a
    name against work they did not do.
    """
    from ..models import Run

    for entry in sorted(recorded, key=lambda r: r.get("index") or 0):
        if not isinstance(entry, dict):
            continue
        span = entry.get("trial_range") or []
        status = entry.get("status") or "done"
        Run.objects.create(
            experiment=exp,
            status="cancelled" if status in _UNFINISHED else status,
            primary_metric=entry.get("primary_metric") or "",
            stopping=entry.get("stopping") or {},
            stopped_by=entry.get("stopped_by") or "",
            events=entry.get("events") or [],
            started_at=parse_datetime(entry["started_at"]) if entry.get("started_at") else None,
            finished_at=parse_datetime(entry["finished_at"]) if entry.get("finished_at") else None,
            trial_seconds=entry.get("trial_seconds"),
            trial_offset=(span[0] - 1) if len(span) == 2 else None,
            trial_count=(span[1] - span[0] + 1) if len(span) == 2 else None,
            error=entry.get("error") or "",
        )


def snapshot_from_experiment(exp: Experiment, *, provenance: bool = False) -> dict:
    """Build a current-format .ihpo snapshot dict from an Experiment row.

    One object per subject, each stating its subject once, and `result` last
    because it dwarfs everything above it. The dataset and model paths point at
    the row's own stored files, or are empty when it has none — a foreign path
    from an imported file is never echoed back out.

    *provenance* fills in what describes how the experiment was made rather
    than what it is: the two digests, the shape of the data, whether the split
    could stratify, the defaults the optimizer's blanks resolved to, the history
    of runs, and the versions behind the numbers. It adds keys to the sections
    below rather than sections of its own. Off by default because the run engine
    and the detail page rebuild through this function on every run and every
    page load, and the dataset fingerprint reads and hashes the file. Export
    turns it on; nothing else needs it.
    """
    dataset = exp.dataset.path if exp.dataset else ""
    model_path = exp.model_file.path if exp.model_file else ""
    snapshot = {
        "format": io.SNAPSHOT_FORMAT,
        "version": dist_version("codesigner"),
        "name": exp.name,
        "seed": exp.seed,
        "dataset": {"filename": Path(dataset).name if dataset else "",
                    "path": dataset},
        "model": {"kind": "file" if model_path else "registry",
                  "name": exp.model_name, "path": model_path},
        "evaluation": evaluation(exp.cv_folds),
        "metrics": {"names": exp.metric_names,
                    "current": exp.current_metric,
                    "original": exp.original_metric},
        "optimizer": {"name": exp.optimizer_name, "params": exp.optimizer_params},
    }
    if provenance:
        _add_provenance(snapshot, exp, dataset, model_path)
    snapshot["result"] = exp.result
    return snapshot


def _add_provenance(snapshot: dict, exp: Experiment, dataset: str, model_path: str) -> None:
    """Fill in what the record needs and reconstruction does not.

    In place, and into the sections that already exist, so the file says each
    thing once: the dataset's digest goes beside the dataset's path rather than
    into a section of its own that names the dataset again.
    """
    if dataset:
        snapshot["dataset"].update(dataset_fingerprint(dataset))
    snapshot["model"].update(
        model_fingerprint(exp.model_name, model_path, exp.env_meta))
    snapshot["evaluation"].update(evaluation(exp.cv_folds, _target(dataset)))
    snapshot["optimizer"]["defaults_used"] = _defaults_used(exp)
    snapshot["runs"] = [_run_record(index, run)
                        for index, run in enumerate(exp.runs.order_by("id"), start=1)]
    snapshot["environment"] = environment()


def _target(dataset_path: str):
    """The target column, for deciding whether the split could stratify."""
    if not dataset_path or not Path(dataset_path).is_file():
        return None
    try:
        _, y = io._load_frame(Path(dataset_path))
    except Exception:  # noqa: BLE001 — an unreadable dataset leaves it unanswered
        return None
    return y


def _defaults_used(exp: Experiment) -> dict:
    """The blanks, and what the installed optimizer filled them with.

    A blank setting means "whatever that component already does", which is the
    right thing to store and useless to read six months later — nobody knows
    what SMAC's random forest uses for its leaf size. This answers the blanks
    and only the blanks: repeating the settings that were actually chosen would
    say the same thing twice, and a reader finding two copies would reasonably
    wonder which one ran.

    The answers are read out of the installed SMAC's own signatures rather than
    copied here, because a copy is a second source of truth that goes stale
    silently. Reconstruction still goes through `params`, so a SMAC that changes
    a default later reproduces the same *request* rather than today's answer to
    it — and `environment.packages.smac` says which version answered.

    The blanks that stay blank belong to the other search strategy, which has no
    default for them because it has no such component: `BlackBoxFacade` has no
    forest to have a tree count of.
    """
    optimizer = OPTIMIZERS.get(exp.optimizer_name)
    params = exp.optimizer_params or {}
    if optimizer is None:
        return {}
    kind = type(optimizer)
    try:
        resolved = kind(**kind.known_params(params)).resolved_params()
    except Exception:  # noqa: BLE001 — a record is never worth a failed export
        return {}
    return {name: value for name, value in resolved.items()
            if params.get(name) is None and value is not None}


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
        "stopping": run.stopping,
        "stopped_by": run.stopped_by or None,
        # The split between trial time and search overhead, and why a run
        # failed. Both are on the row and neither was recorded, which made a
        # round trip lossy the moment the history started being read back.
        "trial_seconds": run.trial_seconds,
        "error": run.error or None,
        "events": run.events or [],
    }


def _dataset_name(snapshot: dict) -> str:
    stored = (snapshot.get("dataset") or {}).get("path", "")
    return Path(stored).name if stored else "dataset.csv"
