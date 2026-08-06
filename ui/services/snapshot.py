"""The single seam between the .ihpo snapshot format and the database.

`experiment_from_snapshot` builds (and saves) an Experiment row from a parsed
snapshot; `snapshot_from_experiment` produces a current-version snapshot dict
from a row. Import, export, and detail-page reconstruction all go through here.
"""

from importlib.metadata import version as dist_version
from pathlib import Path

from django.conf import settings
from django.core.files import File

from ..models import Experiment


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


def snapshot_from_experiment(exp: Experiment) -> dict:
    """Build a current-version .ihpo snapshot dict from an Experiment row.

    dataset_path points at the row's stored file, or is empty when it has none
    — a foreign path from an imported file is never echoed back out.
    """
    return {
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


def _dataset_name(snapshot: dict) -> str:
    stored = snapshot.get("dataset_path", "")
    return Path(stored).name if stored else "dataset.csv"
