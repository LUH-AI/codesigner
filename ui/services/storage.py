"""What a group stores, its limit, and making room.

A group's usage is the bytes its experiments keep on this instance:

* each experiment's dataset — unless it is a bundled demo, which is shared by
  every experiment and counts against nobody;
* an uploaded model's file, and the runner and lock generated beside it;
* the trial models kept for export (`TrialModel`, ui/services/trial_models.py).

Experiments in the bin count until they are purged: their files are all still
there. The model environments are not counted — uv keeps them in one cache
that every experiment's packages share, so none of it is any one group's.

**Datasets come first.** A dataset or model upload that takes a group past its
limit is still accepted, and kept trial models — the stalest first, by when
they were kept or last exported — are deleted until it fits. Only when the
datasets and model files alone exceed the limit is an upload refused. A trial
model that does not fit is simply not kept: parameters never displace data.

An experiment with no group — an instance without accounts — has no limit.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from django.db.models import Sum

#: How long a group's usage is remembered for the banner on every page, in
#: seconds. Long enough that a page load does not walk the disk, short enough
#: that the banner follows an upload or a purge within a page or two.
USAGE_CACHE_SECONDS = 60


def _size(path) -> int:
    try:
        return Path(path).stat().st_size if path and Path(path).is_file() else 0
    except OSError:
        return 0


def _model_files(data):
    """An uploaded model's file and what was generated beside it."""
    if not data.model_file:
        return []
    model = Path(data.model_file.path)
    runner = model.with_name(model.stem + "_runner.py")
    return [model, runner, runner.with_name(runner.name + ".lock")]


def fixed_bytes(exp) -> int:
    """What *exp* stores that is not a kept trial model: its dataset, unless a
    demo, and its model's files."""
    data = exp.data
    total = 0 if data.demo_dataset or not data.dataset else _size(data.dataset.path)
    return total + sum(_size(path) for path in _model_files(data))


def experiment_bytes(exp) -> int:
    """Everything *exp* stores on this instance."""
    kept = exp.trial_models.aggregate(total=Sum("bytes"))["total"] or 0
    return fixed_bytes(exp) + kept


def _group_experiments(group):
    from ..models import Experiment

    return Experiment.objects.filter(group=group).select_related("data")


def usage(group) -> dict:
    """*group*'s storage: ``{"fixed", "models", "total", "limit", "share"}`` —
    datasets and model files, kept trial models, their sum, the limit, and
    the share of it used."""
    from ..models import TrialModel

    experiments = list(_group_experiments(group))
    fixed = sum(fixed_bytes(exp) for exp in experiments)
    models = TrialModel.objects.filter(experiment__group=group).aggregate(
        total=Sum("bytes"))["total"] or 0
    limit = group.storage_limit
    total = fixed + models
    return {"fixed": fixed, "models": models, "total": total, "limit": limit,
            "share": total / limit if limit else 0.0}


def cached_usage(group) -> dict:
    """`usage`, remembered briefly — for what every page shows."""
    key = f"storage-usage-{group.pk}"
    found = cache.get(key)
    if found is None:
        found = usage(group)
        cache.set(key, found, USAGE_CACHE_SECONDS)
    return found


def forget(group) -> None:
    """Drop *group*'s remembered usage, after it changed."""
    if group is not None:
        cache.delete(f"storage-usage-{group.pk}")


def make_room(group, *, keep=None) -> tuple[bool, int]:
    """Bring *group* within its limit by deleting kept trial models, the
    stalest first. Returns (within the limit now, how many were deleted).

    Not within it when its datasets and model files alone are over: those are
    never deleted here. *keep* is a kept model not to delete — the one being
    kept now.
    """
    from ..models import TrialModel
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
    """Whether *group* can store *more* bytes of kept trial models without
    passing its limit. Always, for no group."""
    if group is None:
        return True
    now = usage(group)
    return now["total"] + more <= now["limit"]


def delete_experiment_files(exp) -> None:
    """Delete *exp* for good: its files — dataset (unless a demo, or another
    experiment's too), model and what was generated beside it, kept trial
    models — then its data, which takes the experiment and its runs with it."""
    from ..models import ExperimentData
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
    """The groups *user* belongs to that are past the warning share of their
    limit, with their usage — for the banner on every page."""
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


def human(n: int) -> str:
    """*n* bytes, readably: 1.5 GB, 820 MB."""
    for unit, size in (("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("kB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}".replace(".0 ", " ")
    return f"{n} B"
