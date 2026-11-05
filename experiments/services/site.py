"""`site` names experiments by `identifier`, so a site operator can act without
reading the work.
"""

from datetime import datetime
from typing import cast

from django.db.models import Count, Func, IntegerField, OuterRef, Subquery, Sum
from django.db.models.functions import Coalesce

from access.models import Group, Membership, shown_name
from experiments.models import Experiment, Run


class TrialCount(Func):
    """`TrialCount` counts the trials in a stored result (`result.data`)
    without loading it, so a list of experiments can show each one's count;
    none stored counts 0."""

    output_field = IntegerField()

    def as_sqlite(self, compiler, connection, **extra):
        return self.as_sql(compiler, connection, **extra,
                           template="COALESCE(json_array_length(%(expressions)s, '$.data'), 0)")

    def as_postgresql(self, compiler, connection, **extra):
        return self.as_sql(compiler, connection, **extra,
                           template="COALESCE(jsonb_array_length(%(expressions)s -> 'data'), 0)")


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
    """`discard` deletes a binned experiment for good, with its runs and their
    jobs, through `.bin.purge`."""
    from experiments.services.bin import purge

    purge(experiment)


def active_runs():
    return (Run.objects.filter(status__in=Run.ACTIVE_STATUSES)
            .select_related("experiment", "started_by").order_by("-started_at"))


def job_row(run) -> dict:
    """`job_row` only names the experiment of *run* by `identifier`, and its
    latest job's backend and scheduler number."""
    owner = run.started_by or run.experiment.owner
    group = run.experiment.group
    job = run.job
    return {
        "pk": run.pk,
        "job_id": job.scheduler_id if job else "",
        "backend": job.backend if job else "",
        "status": run.get_status_display(),
        "started_at": run.started_at,
        "duration": run.duration,
        "trial_count": run.trial_count,
        "identifier": run.experiment.identifier,
        "who": shown_name(owner) if owner else "",
        "email": getattr(owner, "email", ""),
        "group": group.name if group else "",
    }


def stop_job(run, reason: str) -> None:
    """`stop_job` cancels the scheduler jobs of *run* and fails *run* with
    *reason*; its jobs end as cancelled, since somebody stopped them."""
    from experiments.services import run as run_service

    run_service.abandon_jobs([run])
    run_service.fail_runs([run], reason, jobs_end="cancelled")


def _window(since=None, until=None) -> dict:
    """`_window` filters runs to those started from *since* to *until*, both
    dates and both included; None leaves that end open."""
    window = {}
    if since is not None:
        window["started_at__date__gte"] = since
    if until is not None:
        window["started_at__date__lte"] = until
    return window


def _run_total(aggregate, *, empty=None, window=None, **scope):
    """`_run_total` aggregates runs in a correlated subquery, so joins cannot
    multiply the total.
    """
    runs = (Run.objects.filter(**scope, **(window or {})).order_by()
            .values(*scope)
            .annotate(value=aggregate)
            .values("value")[:1])
    total = Subquery(runs)
    return Coalesce(total, empty) if empty is not None else total


def usage_by_group(since=None, until=None):
    """`usage_by_group` totals members, and the runs, trials and trial seconds
    of runs started from *since* to *until*, per group."""
    window = _window(since, until)
    return Group.objects.annotate(
        members=Count("memberships", distinct=True),
        runs=_run_total(Count("pk"), empty=0, window=window, experiment__group=OuterRef("pk")),
        trials=_run_total(Sum("trial_count"), window=window, experiment__group=OuterRef("pk")),
        seconds=_run_total(Sum("trial_seconds"), window=window,
                           experiment__group=OuterRef("pk")),
    )


def usage_by_person(since=None, until=None):
    """`usage_by_person` scopes each member's totals to one group, over runs
    started from *since* to *until*."""
    window = _window(since, until)
    return (Membership.objects
            .select_related("user", "group")
            .annotate(**{name: _run_total(agg, empty=empty, window=window,
                                          experiment__owner=OuterRef("user"),
                                          experiment__group=OuterRef("group"))
                         for name, agg, empty in (("runs", Count("pk"), 0),
                                                  ("trials", Sum("trial_count"), None),
                                                  ("seconds", Sum("trial_seconds"), None))})
            .order_by("group__name", "user__username"))


def _live():
    return Experiment.objects.filter(draft=False, trashed_at__isnull=True,
                                     deleted_at__isnull=True)


def usage_by_model(since=None, until=None) -> list[dict]:
    """`usage_by_model` totals, per model, its live experiments, their owners
    and groups, and the trials and trial seconds of runs started from *since*
    to *until*, most seconds first."""
    counts = {row["data__model_name"]: row for row in (
        _live().values("data__model_name")
        .annotate(experiments=Count("pk"), people=Count("owner", distinct=True),
                  groups=Count("group", distinct=True)))}
    runs = {row["experiment__data__model_name"]: row for row in (
        Run.objects.filter(**_window(since, until)).values("experiment__data__model_name")
        .annotate(runs=Count("pk"), trials=Sum("trial_count"), seconds=Sum("trial_seconds")))}
    rows = []
    for model in sorted(set(counts) | set(runs)):
        c, r = counts.get(model, {}), runs.get(model, {})
        rows.append({"model": model, "experiments": c.get("experiments", 0),
                     "people": c.get("people", 0), "groups": c.get("groups", 0),
                     "trials": r.get("trials") or 0, "runs": r.get("runs", 0),
                     "seconds": r.get("seconds") or 0.0})
    return sorted(rows, key=lambda row: row["seconds"], reverse=True)


def model_seconds_by_group(since=None, until=None) -> list[dict]:
    """`model_seconds_by_group` gives trials, runs and trial seconds per model
    and group, over runs started from *since* to *until*."""
    return [{"model": row["experiment__data__model_name"],
             "group": row["experiment__group__name"] or "",
             "trials": row["trials"] or 0, "runs": row["runs"],
             "seconds": row["seconds"] or 0.0}
            for row in (Run.objects.filter(**_window(since, until))
                        .values("experiment__data__model_name", "experiment__group__name")
                        .annotate(trials=Sum("trial_count"), runs=Count("pk"),
                                  seconds=Sum("trial_seconds"))
                        .order_by("experiment__data__model_name", "experiment__group__name"))]
