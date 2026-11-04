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
