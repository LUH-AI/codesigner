"""Site-wide operations on experiments and runs: the site bin, active jobs and
usage totals. They identify experiments by `identifier` rather than name, so a
site operator can act without reading the work.
"""

from datetime import datetime
from typing import cast

from django.db.models import Count, OuterRef, Subquery, Sum
from django.db.models.functions import Coalesce

from access.models import Group, Membership
from experiments.models import Experiment, Run


def trash():
    """`trash` returns the site bin: a deleted group's work, plus deletions no
    person's bin still contains (`.orphaned`)."""
    from experiments.services.bin import orphaned

    group_gone = Experiment.objects.filter(trashed_at__isnull=False, deleted_at__isnull=True)
    return (group_gone | orphaned()).distinct()


def trash_rows():
    """`trash_rows` lists the site bin newest first, with each experiment's
    former owner and run count."""
    rows = trash().select_related("owner").annotate(run_count=Count("runs", distinct=True))
    return sorted(rows, key=lambda e: cast(datetime, e.deleted_at or e.trashed_at), reverse=True)


def rehome(experiment, group) -> None:
    """`rehome` puts a binned experiment back into *group*, which decides who
    may reach it; the owner stays as is, even when absent."""
    from experiments.services.bin import put_back

    put_back(experiment, group)


def discard(experiment) -> None:
    """`discard` destroys a binned experiment and its files for good."""
    from experiments.services.storage import delete_experiment_files

    delete_experiment_files(experiment)


def active_runs():
    """`active_runs` returns the runs in progress across every group, newest
    first."""
    return (Run.objects.filter(status__in=Run.ACTIVE_STATUSES)
            .select_related("experiment", "started_by").order_by("-started_at"))


def job_row(run) -> dict:
    """`job_row` describes active *run* for a site operator: its job, who
    started it and the experiment's group, by identifier only."""
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
    """`stop_job` stops active *run* forcibly, cancelling its cluster job
    too, and records *reason* as its error."""
    from experiments.services import run as run_service

    if run.job_id:
        run_service.abandon_cluster_job(run.job_id)
    run_service.fail_runs([run], reason)


def _run_total(aggregate, *, empty=None, **scope):
    """`_run_total` aggregates runs in a correlated subquery so joins elsewhere
    cannot multiply the total. A group without runs gets *empty* in place of
    null, and counts then read 0.
    """
    runs = (Run.objects.filter(**scope).order_by()
            .values(*scope)
            .annotate(value=aggregate)
            .values("value")[:1])
    total = Subquery(runs)
    return Coalesce(total, empty) if empty is not None else total


def usage_by_group():
    """`usage_by_group` totals each group's members, runs, trials and trial
    seconds. Grouping by `.Experiment.group` counts each run once."""
    return Group.objects.annotate(
        members=Count("memberships", distinct=True),
        runs=_run_total(Count("pk"), empty=0, experiment__group=OuterRef("pk")),
        trials=_run_total(Sum("trial_count"), experiment__group=OuterRef("pk")),
        seconds=_run_total(Sum("trial_seconds"), experiment__group=OuterRef("pk")),
    )


def usage_by_person():
    """`usage_by_person` gives per-account totals within each group. Scoping
    to the row's group keeps a member's totals from repeating under every
    group."""
    return (Membership.objects
            .select_related("user", "group")
            .annotate(**{name: _run_total(agg, empty=empty,
                                          experiment__owner=OuterRef("user"),
                                          experiment__group=OuterRef("group"))
                         for name, agg, empty in (("runs", Count("pk"), 0),
                                                  ("trials", Sum("trial_count"), None),
                                                  ("seconds", Sum("trial_seconds"), None))})
            .order_by("group__name", "user__username"))
