"""A group's usage counts datasets (except bundled demos), uploaded model files
with their runner and lock, and kept trial models. Binned experiments count
until purged, and uv's shared environment cache belongs to no group.

An upload over the limit deletes the stalest kept trial models, and the upload
only fails when datasets and model files alone exceed the limit. An experiment
without a group has no limit.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from django.db.models import Sum

#: Seconds; spares page loads a disk walk.
USAGE_CACHE_SECONDS = 60


def _size(path) -> int:
    try:
        return Path(path).stat().st_size if path and Path(path).is_file() else 0
    except OSError:
        return 0


def _model_files(data):
    """`_model_files` lists the model, runner and lock."""
    if not data.model_file:
        return []
    model = Path(data.model_file.path)
    runner = model.with_name(model.stem + "_runner.py")
    return [model, runner, runner.with_name(runner.name + ".lock")]


def fixed_bytes(exp) -> int:
    """`fixed_bytes` counts the dataset (unless a demo) and model files."""
    data = exp.data
    total = 0 if data.demo_dataset or not data.dataset else _size(data.dataset.path)
    return total + sum(_size(path) for path in _model_files(data))


def experiment_bytes(exp) -> int:
    kept = exp.trial_models.aggregate(total=Sum("bytes"))["total"] or 0
    return fixed_bytes(exp) + kept


def _group_experiments(group):
    from experiments.models import Experiment

    return Experiment.objects.filter(group=group).select_related("data")


def usage(group) -> dict:
    """`usage` returns the keys fixed, models, total, limit and share."""
    from experiments.models import TrialModel

    experiments = list(_group_experiments(group))
    fixed = sum(fixed_bytes(exp) for exp in experiments)
    models = TrialModel.objects.filter(experiment__group=group).aggregate(
        total=Sum("bytes"))["total"] or 0
    limit = group.storage_limit
    total = fixed + models
    return {"fixed": fixed, "models": models, "total": total, "limit": limit,
            "share": total / limit if limit else 0.0}


def cached_usage(group) -> dict:
    key = f"storage-usage-{group.pk}"
    found = cache.get(key)
    if found is None:
        found = usage(group)
        cache.set(key, found, USAGE_CACHE_SECONDS)
    return found


def forget(group) -> None:
    if group is not None:
        cache.delete(f"storage-usage-{group.pk}")


def make_room(group, *, keep=None) -> tuple[bool, int]:
    """`make_room` deletes the stalest kept trial models and returns (within the
    limit, how many went). The datasets and model files stay, and *keep*
    protects the model being kept now.
    """
    from experiments.models import TrialModel
    from . import trial_models

    if group is None:
        return True, 0
    now = usage(group)
    deleted = 0
    if now["total"] > now["limit"]:
        over = now["total"] - now["limit"]
        stale = (TrialModel.objects.filter(experiment__group=group)
                 .exclude(pk=getattr(keep, "pk", None)).order_by("last_used_at", "pk"))
        for kept in stale:
            if over <= 0:
                break
            over -= kept.bytes
            trial_models.discard(kept)
            deleted += 1
    forget(group)
    return usage(group)["total"] <= group.storage_limit, deleted


def has_room(group, more: int) -> bool:
    """`has_room` returns True for no group."""
    if group is None:
        return True
    now = usage(group)
    return now["total"] + more <= now["limit"]


def delete_experiment_files(exp) -> None:
    """The dataset survives when a demo or shared with another experiment. The
    data row's deletion cascades to the experiment and its runs."""
    from experiments.models import ExperimentData
    from . import trial_models

    data, group = exp.data, exp.group
    for kept in exp.trial_models.all():
        trial_models.discard(kept)
    paths = list(_model_files(data))
    if data.dataset and not data.demo_dataset and not ExperimentData.objects.filter(
            dataset=data.dataset.name).exclude(pk=data.pk).exists():
        paths.append(Path(data.dataset.path))
    data.delete()
    for path in paths:
        Path(path).unlink(missing_ok=True)
    forget(group)


def warnings_for(user) -> list[dict]:
    """`warnings_for` lists groups past `STORAGE_WARNING_SHARE` for the page
    banner."""
    if not getattr(user, "is_authenticated", False):
        return []
    from access.models import Group

    share = settings.STORAGE_WARNING_SHARE
    out = []
    for group in Group.objects.filter(memberships__user=user).distinct():
        now = cached_usage(group)
        if now["share"] >= share:
            out.append({"group": group, **now})
    return out
