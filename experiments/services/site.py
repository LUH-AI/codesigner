"""`site` names experiments by `identifier`, so a site operator can act without
reading the work.
"""

from datetime import datetime
from typing import cast

from django.db.models import Count, OuterRef, Subquery, Sum
from django.db.models.functions import Coalesce

from access.models import Group, Membership
from experiments.models import Experiment, Run


def trash():
    """`trash` returns a deleted group's work plus `.orphaned` deletions."""
    from experiments.services.bin import orphaned

    group_gone = Experiment.objects.filter(trashed_at__isnull=False, deleted_at__isnull=True)
    return (group_gone | orphaned()).distinct()


def trash_rows():
    """`trash_rows` sorts `trash` newest first, with owner and run count."""
    rows = trash().select_related("owner").annotate(run_count=Count("runs", distinct=True))
    return sorted(rows, key=lambda e: cast(datetime, e.deleted_at or e.trashed_at), reverse=True)


def rehome(experiment, group) -> None:
    """`rehome` moves a binned experiment into *group* and keeps its owner."""
    from experiments.services.bin import put_back

    put_back(experiment, group)


def discard(experiment) -> None:
    """`discard` deletes a binned experiment for good."""
    from experiments.services.storage import delete_experiment_files

    delete_experiment_files(experiment)


def active_runs():
    return (Run.objects.filter(status__in=Run.ACTIVE_STATUSES)
            .select_related("experiment", "started_by").order_by("-started_at"))


def job_row(run) -> dict:
    """`job_row` only names the experiment of *run* by `identifier`."""
    owner = run.started_by or run.experiment.owner
    group = run.experiment.group
    return {
        "pk": run.pk,
        "job_id": run.job_id,
        "backend": run.backend,
        "status": run.get_status_display(),
        "started_at": run.started_at,
        "duration": run.duration,
        "trial_count": run.trial_count,
        "identifier": run.experiment.identifier,
        "who": owner.get_username() if owner else "",
        "email": getattr(owner, "email", ""),
        "group": group.name if group else "",
    }


def stop_job(run, reason: str) -> None:
    """`stop_job` cancels the cluster job of *run* and fails *run* with
    *reason*."""
    from experiments.services import run as run_service

    if run.job_id:
        run_service.abandon_cluster_job(run.job_id)
    run_service.fail_runs([run], reason)


def _run_total(aggregate, *, empty=None, **scope):
    """`_run_total` aggregates runs in a correlated subquery, so joins cannot
    multiply the total.
    """
    runs = (Run.objects.filter(**scope).order_by()
            .values(*scope)
            .annotate(value=aggregate)
            .values("value")[:1])
    total = Subquery(runs)
    return Coalesce(total, empty) if empty is not None else total


def usage_by_group():
    """`usage_by_group` totals members, runs, trials and trial seconds per
    group."""
    return Group.objects.annotate(
        members=Count("memberships", distinct=True),
        runs=_run_total(Count("pk"), empty=0, experiment__group=OuterRef("pk")),
        trials=_run_total(Sum("trial_count"), experiment__group=OuterRef("pk")),
        seconds=_run_total(Sum("trial_seconds"), experiment__group=OuterRef("pk")),
    )


def usage_by_person():
    """`usage_by_person` scopes each member's totals to one group."""
    return (Membership.objects
            .select_related("user", "group")
            .annotate(**{name: _run_total(agg, empty=empty,
                                          experiment__owner=OuterRef("user"),
                                          experiment__group=OuterRef("group"))
                         for name, agg, empty in (("runs", Count("pk"), 0),
                                                  ("trials", Sum("trial_count"), None),
                                                  ("seconds", Sum("trial_seconds"), None))})
            .order_by("group__name", "user__username"))


def _live():
    return Experiment.objects.filter(draft=False, trashed_at__isnull=True,
                                     deleted_at__isnull=True)


def usage_by_model() -> list[dict]:
    """`usage_by_model` totals, per model, its live experiments, their owners
    and groups, and every run's trials and trial seconds, most seconds first."""
    counts = {row["data__model_name"]: row for row in (
        _live().values("data__model_name")
        .annotate(experiments=Count("pk"), people=Count("owner", distinct=True),
                  groups=Count("group", distinct=True)))}
    runs = {row["experiment__data__model_name"]: row for row in (
        Run.objects.values("experiment__data__model_name")
        .annotate(runs=Count("pk"), trials=Sum("trial_count"), seconds=Sum("trial_seconds")))}
    rows = []
    for model in sorted(set(counts) | set(runs)):
        c, r = counts.get(model, {}), runs.get(model, {})
        rows.append({"model": model, "experiments": c.get("experiments", 0),
                     "people": c.get("people", 0), "groups": c.get("groups", 0),
                     "runs": r.get("runs", 0), "trials": r.get("trials") or 0,
                     "seconds": r.get("seconds") or 0.0})
    return sorted(rows, key=lambda row: row["seconds"], reverse=True)


def model_seconds_by_group() -> list[dict]:
    """`model_seconds_by_group` gives trial seconds and runs per model and
    group."""
    return [{"model": row["experiment__data__model_name"],
             "group": row["experiment__group__name"] or "",
             "runs": row["runs"], "seconds": row["seconds"] or 0.0}
            for row in (Run.objects
                        .values("experiment__data__model_name", "experiment__group__name")
                        .annotate(runs=Count("pk"), seconds=Sum("trial_seconds"))
                        .order_by("experiment__data__model_name", "experiment__group__name"))]


def _quantile(ordered: list, q: float) -> float:
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def best_configurations() -> dict:
    """`best_configurations` gathers, per model, the best configuration of
    every live experiment with a finished search: for a numeric
    hyperparameter the spread of its best values within the range searched,
    for a categorical one how often each choice won, and for both the mean
    importance the experiments measured. Hyperparameters come most important
    first.
    """
    import math

    gathered: dict = {}
    for data in (exp.data for exp in _live().filter(data__result__isnull=False)
                 .select_related("data")):
        result = data.result or {}
        best = (result.get("configs") or {}).get(str(result.get("best_config_id")))
        if not best:
            continue
        space = {hp["name"]: hp for hp in (data.config_space or {}).get("hyperparameters", [])}
        importance = (result.get("hyperparameter_importance") or {}).get(
            result.get("primary_metric")) or {}
        model = gathered.setdefault(data.model_name, {"experiments": 0, "hps": {}})
        model["experiments"] += 1
        for name, value in best.items():
            hp = model["hps"].setdefault(name, {"values": [], "importance": [],
                                                "lower": None, "upper": None, "log": False})
            hp["values"].append(value)
            if isinstance(importance.get(name), (int, float)):
                hp["importance"].append(importance[name])
            bounds = space.get(name, {})
            if "lower" in bounds:
                hp["lower"] = bounds["lower"] if hp["lower"] is None else min(hp["lower"], bounds["lower"])
                hp["upper"] = bounds["upper"] if hp["upper"] is None else max(hp["upper"], bounds["upper"])
                hp["log"] = hp["log"] or bool(bounds.get("log"))

    def position(hp, x):
        low, high = hp["lower"], hp["upper"]
        if low is None or high is None or high <= low:
            return None
        if hp["log"] and low > 0 and x > 0:
            return (math.log(x) - math.log(low)) / (math.log(high) - math.log(low))
        return (x - low) / (high - low)

    summary = {}
    for model, found in gathered.items():
        rows = []
        for name, hp in found["hps"].items():
            numbers = [v for v in hp["values"] if isinstance(v, (int, float))
                       and not isinstance(v, bool)]
            row = {"name": name, "n": len(hp["values"]),
                   "importance": (sum(hp["importance"]) / len(hp["importance"])
                                  if hp["importance"] else None)}
            if numbers and len(numbers) == len(hp["values"]):
                ordered = sorted(numbers)
                stats = {"min": ordered[0], "q1": _quantile(ordered, 0.25),
                         "median": _quantile(ordered, 0.5), "q3": _quantile(ordered, 0.75),
                         "max": ordered[-1]}
                row.update(kind="numeric", lower=hp["lower"], upper=hp["upper"], log=hp["log"],
                           **stats, positions={k: position(hp, v) for k, v in stats.items()})
            else:
                counts: dict = {}
                for value in hp["values"]:
                    counts[str(value)] = counts.get(str(value), 0) + 1
                row.update(kind="categorical", choices=sorted(
                    ({"choice": c, "share": n / len(hp["values"])} for c, n in counts.items()),
                    key=lambda choice: choice["share"], reverse=True))
            rows.append(row)
        rows.sort(key=lambda row: (row["importance"] is None, -(row["importance"] or 0)))
        summary[model] = {"experiments": found["experiments"], "hyperparameters": rows}
    return summary
