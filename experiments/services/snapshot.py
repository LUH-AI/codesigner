"""`experiment_from_snapshot` and `snapshot_from_experiment` are the only bridge
between the `.ihpo` format and the database.
"""

import copy
from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from core import datasets, io, tasks
from core.processing import choices_of
from core.provenance import (
    dataset_fingerprint, environment, evaluation, model_fingerprint, sha256,
)
from core.registry import METRICS, MODELS, OPTIMIZERS, canonical_model_name
from experiments.models import Experiment, ExperimentData, Run

from . import history, modelenv, timestamps


def finish_import(exp, *, source: str, filename: str, user=None) -> None:
    """`finish_import` re-locks an imported model instead of trusting the
    exporter's pins."""
    history.record(exp, "imported", user=user, source=source, filename=filename,
                   dataset_attached=exp.data.has_dataset)
    modelenv.start_preparation(exp)


def experiment_from_snapshot(snapshot: dict, dataset_file=None, model_file=None,
                             adopt_paths: bool = False, owner=None,
                             group=None) -> Experiment:
    """`experiment_from_snapshot` adopts files from *dataset_file* and
    *model_file*, and without them the row stays browsable until someone
    attaches a dataset. *adopt_paths* lets the server read the snapshot's paths,
    so only the create form and the import command, which wrote those paths,
    enable it. `experiment_from_snapshot` adopts a custom model only when
    ALLOW_CUSTOM_MODELS is on, and takes *owner* because an `.ihpo` records no
    author.
    """
    snapshot = timestamps.anchor(copy.deepcopy(io.normalize(snapshot)))
    data = data_from_snapshot(snapshot, dataset_file, model_file, adopt_paths)

    # The transaction writes both halves or neither, so no orphan
    # `ExperimentData` row survives a failure.
    with transaction.atomic():
        data.save()
        exp = Experiment(data=data, owner=owner, group=group)
        exp.save()
        _restore_runs(exp, snapshot.get("runs") or [])
        history.restore(exp, snapshot.get("history"))
    return exp


def data_from_snapshot(snapshot: dict, dataset_file=None, model_file=None,
                       adopt_paths: bool = False) -> ExperimentData:
    """`data_from_snapshot` returns the unsaved portable half with its files
    saved; a draft's setup reuses it at each save of step 1."""
    ev = io.evaluation_of(snapshot)
    data = ExperimentData(
        name=snapshot.get("name") or "",
        # A renamed built-in takes its current name; an upload keeps its own.
        model_name=(snapshot["model"]["name"] if snapshot["model"].get("kind") == "file"
                    else canonical_model_name(snapshot["model"]["name"])),
        optimizer_name=snapshot["optimizer"]["name"],
        optimizer_params=snapshot["optimizer"].get("params") or {},
        metric_names=snapshot["metrics"]["names"],
        current_metric=snapshot["metrics"].get("current"),
        original_metric=snapshot["metrics"].get("original"),
        seed=snapshot["seed"],
        created_at=_when(snapshot.get("began_at")) or timezone.now(),
        time_basis=timestamps.basis_of(snapshot),
        cv_folds=ev["folds"],
        test_size=ev["test_size"],
        time_column=ev["time_column"],
        time_gap=ev["gap"],
        horizon=ev["horizon"],
        series_column=ev["series_column"],
        season=ev["season"],
        task=tasks.task_of(snapshot),
        processing=choices_of((snapshot.get("dataset") or {}).get("processing")),
        config_space=snapshot.get("space"),
        priors=snapshot.get("priors") or {},
        result=snapshot.get("result"),
    )

    demo = _demo_named(snapshot, adopt_paths)
    if dataset_file is not None:
        name = getattr(dataset_file, "name", None) or _dataset_name(snapshot)
        data.dataset.save(Path(name).name, dataset_file, save=False)
    elif demo:
        data.demo_dataset = demo
    elif adopt_paths:
        stored = snapshot["dataset"].get("path", "")
        if stored and Path(stored).is_file():
            with open(stored, "rb") as fh:
                data.dataset.save(Path(stored).name, File(fh), save=False)

    if model_file is not None:
        name = getattr(model_file, "name", None) or "model.py"
        data.model_file.save(Path(name).name, model_file, save=False)
    elif adopt_paths and settings.ALLOW_CUSTOM_MODELS:
        stored_model = snapshot["model"].get("path", "")
        if stored_model and Path(stored_model).is_file():
            with open(stored_model, "rb") as fh:
                data.model_file.save(Path(stored_model).name, File(fh), save=False)

    return data


#: An unfinished run was interrupted, so it imports as cancelled instead of
#: polling forever.
_UNFINISHED = Run.ACTIVE_STATUSES


def _restore_runs(exp: Experiment, recorded: list) -> None:
    """`_restore_runs` keeps the round trip lossless for the `runs` section.
    `started_by` stays empty, because an account elsewhere has no local
    counterpart.
    """
    from experiments.models import Run

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
            priors=entry.get("priors") or {},
            configurations=entry.get("configurations") or [],
            trial_timeout=entry.get("trial_timeout") or {},
            created_at=_when(entry.get("created_at")),
            started_at=_when(entry.get("started_at")),
            finished_at=_when(entry.get("finished_at")),
            cancel_requested=bool(entry.get("cancel_requested_at")),
            cancel_requested_at=_when(entry.get("cancel_requested_at")),
            trial_seconds=entry.get("trial_seconds"),
            trial_offset=(span[0] - 1) if len(span) == 2 else None,
            trial_count=(span[1] - span[0] + 1) if len(span) == 2 else None,
            error=entry.get("error") or "",
        )


def _when(value):
    return parse_datetime(value) if value else None


def _model_kind(data, model_path: str) -> str:
    """`_model_kind` infers the kind: a `.py` is a file, a registry name is a
    registry model, and anything else is external, such as a SMAC import.
    """
    if model_path:
        return "file"
    return "registry" if data.model_name in MODELS else "external"


def _demo_named(snapshot, adopt_paths) -> str:
    """`_demo_named` matches a bundled demo by name, or by file for a path the
    caller produced."""
    section = snapshot.get("dataset") or {}
    demos = datasets.demo_datasets()
    named = section.get("demo")
    if named and named in demos:
        return named
    stored = section.get("path", "")
    if adopt_paths and stored:
        resolved = Path(stored).resolve()
        return next((name for name, path in demos.items() if Path(path).resolve() == resolved), "")
    return ""


def snapshot_from_experiment(exp: Experiment, *, provenance: bool = False) -> dict:
    """`snapshot_from_experiment` points paths at the row's own files and never
    echoes a foreign path.

    *provenance* adds digests, data shape, resolved defaults, run history and
    versions; export alone enables it because the fingerprint hashes the
    dataset.
    """
    # Everything except the last two lines reads `data`, so `ExperimentData`
    # defines the file format.
    data = exp.data
    dataset = str(data.dataset_path() or "")
    model_path = data.model_file.path if data.model_file else ""
    snapshot = {
        "format": io.SNAPSHOT_FORMAT,
        "version": dist_version("codesigner"),
        "name": data.name,
        "seed": data.seed,
        # Start time and time form; an export restates the form.
        "began_at": data.created_at.isoformat() if data.created_at else None,
        "timestamps": data.time_basis,
        "dataset": {"filename": Path(dataset).name if dataset else "",
                    "path": dataset,
                    # Another instance reads a demo by name from its own copy.
                    "demo": data.demo_dataset or None,
                    "processing": choices_of(data.processing)},
        "model": {"kind": _model_kind(data, model_path),
                  "name": data.model_name, "path": model_path},
        "evaluation": evaluation(data.cv_folds, test_size=data.test_size, task=data.task,
                                 time_column=data.time_column, gap=data.time_gap,
                                 horizon=data.horizon, series_column=data.series_column,
                                 season=data.season),
        "metrics": {"names": data.metric_names,
                    "current": data.current_metric,
                    "original": data.original_metric},
        "optimizer": {"name": data.optimizer_name, "params": data.optimizer_params},
    }
    # The `space` section only appears when the row stores a search space.
    if data.config_space:
        snapshot["space"] = data.config_space
    # `priors` sits beside `space` and steers the next run, so a re-import keeps
    # searching weighted. The section is optional, so `format` stays unchanged.
    if data.priors:
        snapshot["priors"] = {name: _prior_record(stated)
                              for name, stated in sorted(data.priors.items())}
    if provenance:
        _add_provenance(snapshot, exp, dataset, model_path)
    snapshot["result"] = data.result
    return snapshot


#: A prior is its shape, params, decay and stating time. `knots` stays out
#: because the grid density would restate `kind` and `params` and could disagree
#: with them.
_PRIOR_FIELDS = ("kind", "params", "decay", "at_trial", "delay_decay")


def export(exp: Experiment, mode: str, keep_tracebacks: bool) -> bytes:
    """`export` writes *exp* as an `.ihpo` with times in *mode*, keeping
    tracebacks only when *keep_tracebacks* is set."""
    snapshot = snapshot_from_experiment(exp, provenance=True)
    # Server paths are useless to the file's reader and reveal the instance
    # layout.
    snapshot["dataset"]["path"] = ""
    snapshot["model"]["path"] = ""
    if snapshot.get("result"):
        snapshot["result"] = copy.deepcopy(snapshot["result"])
    timestamps.apply(snapshot, timestamps.clamp(exp, mode), timestamps.origin(exp))
    if not keep_tracebacks and snapshot.get("result"):
        for entry in snapshot["result"].get("data", []):
            # The reason stays; the traceback describes the machine.
            (entry.get("additional_info") or {}).pop("traceback", None)
    return io.to_bytes(snapshot)


def _prior_record(stated: dict) -> dict:
    """`_prior_record` uses an allow-list to drop `exponent`, which is
    recomputed on every request and would contradict the schedule. `delay_decay`
    stays because nothing else records that the belief starts after the initial
    design of the reading run.
    """
    return {key: stated[key] for key in _PRIOR_FIELDS if key in stated}


def _add_provenance(snapshot: dict, exp: Experiment, dataset: str, model_path: str) -> None:
    """Provenance fills existing sections in place, so each fact appears once.
    """
    if dataset:
        snapshot["dataset"].update(dataset_fingerprint(dataset))
    # `env_meta` describes the environment built locally and rides along as
    # provenance.
    data = exp.data
    snapshot["model"].update(
        model_fingerprint(data.model_name, model_path, exp.env_meta,
                          kind=snapshot["model"]["kind"]))
    snapshot["evaluation"].update(
        evaluation(data.cv_folds, _target(dataset), test_size=data.test_size, task=data.task,
                   time_column=data.time_column, gap=data.time_gap, horizon=data.horizon,
                   series_column=data.series_column, season=data.season))
    snapshot["optimizer"]["defaults_used"] = _defaults_used(data)
    snapshot["runs"] = [_run_record(index, run)
                        for index, run in enumerate(exp.runs.order_by("id"), start=1)]
    snapshot["history"] = history.serialized(exp)
    snapshot["environment"] = environment()


def _target(dataset_path: str):
    if not dataset_path or not Path(dataset_path).is_file():
        return None
    try:
        _, y = datasets.load_frame(Path(dataset_path))
    except Exception:  # noqa: BLE001 — an unreadable dataset leaves it unanswered
        return None
    return y


def _defaults_used(data) -> dict:
    """`_defaults_used` records what the installed SMAC filled into blank
    settings, read from its signatures. Reconstruction still uses `params`, and
    blanks of the other strategy stay blank.
    """
    kind = OPTIMIZERS.get(data.optimizer_name)
    params = data.optimizer_params or {}
    if kind is None:
        return {}
    try:
        resolved = kind(**kind.known_params(params)).resolved_params()
    except Exception:  # noqa: BLE001 — a record is never worth a failed export
        return {}
    return {name: value for name, value in resolved.items()
            if params.get(name) is None and value is not None}


def _run_record(index: int, run) -> dict:
    offset = run.trial_offset
    count = run.trial_count
    span = None
    if offset is not None and count is not None:
        # Inclusive 1-based range; empty is [offset + 1, offset].
        span = [offset + 1, offset + count]
    return {
        "index": index,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "cancel_requested_at": (run.cancel_requested_at.isoformat()
                                if run.cancel_requested_at else None),
        "trial_range": span,
        "primary_metric": run.primary_metric,
        "stopping": run.stopping,
        "stopped_by": run.stopped_by or None,
        # The round trip keeps trial time and the failure reason.
        "trial_seconds": run.trial_seconds,
        "error": run.error or None,
        "events": run.events or [],
        "priors": run.priors or {},
        "trial_timeout": run.trial_timeout or {},
        "configurations": run.configurations or [],
    }


def _dataset_name(snapshot: dict) -> str:
    stored = (snapshot.get("dataset") or {}).get("path", "")
    return Path(stored).name if stored else "dataset.csv"


def replace_data(exp: Experiment, new: ExperimentData) -> set[str]:
    """`replace_data` keeps processing, plus priors and space unless the model
    changed."""
    old = exp.data
    changed = what_changed(old, new)
    new.processing = old.processing
    new.priors = {} if "model" in changed else (old.priors or {})
    new.config_space = None if "model" in changed else old.config_space
    with transaction.atomic():
        new.save()
        exp.data = new
        exp.save()
        for field in (old.dataset, old.model_file):
            if field:
                field.delete(save=False)
        old.delete()
    return changed


def what_changed(before, after) -> set[str]:
    def dataset(data):
        if data.demo_dataset:
            return ("demo", data.demo_dataset)
        path = data.dataset_path()
        return ("file", sha256(path)) if path and path.is_file() else None

    def model(data):
        if data.model_file:
            try:
                with data.model_file.open("rb") as f:
                    return ("file", f.read())
            except OSError:
                return ("file", data.model_file.name)
        return ("registry", data.model_name)

    changed = set()
    if dataset(before) != dataset(after):
        changed.add("dataset")
    if before.task != after.task:
        changed.add("task")
    if model(before) != model(after):
        changed.add("model")
    return changed


def rebuild_experiment(exp) -> dict | None:
    """`rebuild_experiment` returns the read-only `build_experiment` dict, or
    None when a metric or optimizer is gone. `built["model"]` is None only for
    custom uploads.
    """
    try:
        _, built = io.build_experiment(
            snapshot_from_experiment(exp),
            METRICS, MODELS, OPTIMIZERS, read_only=True,
        )
    except ValueError:
        return None
    return built


def rebuild_result(exp):
    """`rebuild_result` returns the read-only result, or None."""
    built = rebuild_experiment(exp)
    return built["result"] if built else None


def config_space_for(built):
    """The stored `space` comes first and the model second, since an external
    run has no model and a read-only upload has no reachable model. Figures
    handle None with linear axes or a caption.
    """
    built = built or {}
    stored = io.config_space_from_serialized(built.get("config_space"),
                                             seed=built.get("seed", 0))
    if stored is not None:
        return stored
    model = built.get("model")
    return model.get_config_space(seed=built["seed"]) if model is not None else None
