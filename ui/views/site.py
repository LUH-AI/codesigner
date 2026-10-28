"""A site admin reaches no experiment, so the site views skip
`@experiment_view`.
"""

from datetime import datetime
from functools import wraps
from typing import cast

from django.contrib import messages
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count, OuterRef, Subquery, Sum
from django.db.models.deletion import ProtectedError
from django.db.models.functions import Coalesce
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.models import Group, Membership
from access.policy import is_site_admin, policy
from access.services.accounts import find_or_create_by_email, normalize
from experiments.models import Run
from experiments.services.storage import delete_experiment_files
from experiments.services.storage import forget as forget_storage
from experiments.services.storage import usage as storage_usage
from experiments.services import bin as bin_service
from experiments.services import run as run_service

from ..formatting import human


def site_admin_required(view):
    """`site_admin_required` demands `manage_site`, a permission rather than
    superuser."""
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not is_site_admin(request.user):
            raise Http404
        return view(request, *args, **kwargs)
    return wrapped


@site_admin_required
def site_groups(request):
    """`site_groups` lists every group's size, storage and status, and shows no
    experiments.
    """
    groups = list(Group.objects.annotate(members=Count("memberships")))
    for group in groups:
        now = storage_usage(group)
        setattr(group, "storage_used", human(now["total"]))
        setattr(group, "storage_limit_gb", round(group.storage_limit / 1024 ** 3, 2))
        setattr(group, "storage_percent", round(100 * now["share"]))
    return render(request, "ui/panels/site_groups.html", {"groups": groups})


@require_POST
@site_admin_required
def site_group_save(request):
    """`site_group_save` creates a group or edits one."""
    pk = request.POST.get("pk")
    name = (request.POST.get("name") or "").strip()
    try:
        limit = int(request.POST.get("user_limit") or 0)
    except ValueError:
        limit = -1
    try:
        storage_gb = float(request.POST.get("storage_gb") or "nan")
    except ValueError:
        storage_gb = -1.0

    if not name or limit < 0 or storage_gb < 0:
        messages.error(request, _("A group needs a name, a seat limit of zero or more, and a "
                                  "storage limit of zero or more."))
        return redirect("ui:site_groups")

    if pk:
        group = get_object_or_404(Group, pk=pk)
        group.name = name
        group.user_limit = limit
        group.is_active = bool(request.POST.get("is_active"))
        if storage_gb == storage_gb:  # NaN means left blank
            group.storage_limit_bytes = int(storage_gb * 1024 ** 3)
        group.save(update_fields=["name", "user_limit", "is_active", "storage_limit_bytes"])
        forget_storage(group)
        messages.success(request, _("Saved %(name)s.") % {"name": group.name})
    elif Group.objects.filter(name=name).exists():
        messages.error(request, _("There is already a group called “%(name)s”.")
                       % {"name": name})
    else:
        _create_group(request, name, limit)
    return redirect("ui:site_groups")


def _create_group(request, name, limit):
    """`_create_group` creates the group and its primary lead in one
    transaction, since a group without one could never appoint leads. The lead
    takes a seat, which requires a seat limit of at least 1.
    """

    address = normalize(request.POST.get("primary_lead_email"))
    if not address:
        messages.error(request, _("A new group needs a primary lead's email address."))
        return
    if limit < 1:
        messages.error(request, _("A new group needs at least one seat, for its primary lead."))
        return
    try:
        with transaction.atomic():
            group = Group.objects.create(name=name, user_limit=limit)
            user, created, given = find_or_create_by_email(
                address, password=request.POST.get("password") or "")
            Membership.objects.create(user=user, group=group,
                                      role=Membership.PRIMARY_LEAD)
    except ValidationError:
        messages.error(request, _("“%(email)s” is not an email address.")
                       % {"email": address})
        return
    if created:
        messages.success(request, _(
            "Created %(name)s, and an account for %(email)s as its primary lead. "
            "Its first password is %(password)s — hand it over.") % {
                "name": name, "email": address, "password": given})
    else:
        messages.success(request, _("Created %(name)s, with %(email)s as its primary lead.")
                         % {"name": name, "email": address})


@require_POST
@site_admin_required
def site_group_delete(request):
    """`site_group_delete` moves the group's experiments to the bin via
    `Group.delete()`. `Membership.group` is `PROTECT`, so a group with members
    is refused.
    """
    group = get_object_or_404(Group, pk=request.POST.get("pk") or 0)
    name = group.name
    try:
        group.delete()
    except ProtectedError:
        messages.error(request, _(
            "%(name)s still has people in it. Remove them first.") % {"name": name})
    else:
        messages.success(request, _("Deleted %(name)s. Anything it held is in "
                                    "the bin.") % {"name": name})
    return redirect("ui:site_groups")


@site_admin_required
def site_trash(request):
    """`site_trash` lists the bin by identifier, former owner and cost, without
    names, so the admin can rehome or discard without reading the work.
    """
    trashed = (policy().trash(request)
               .select_related("owner")
               .annotate(run_count=Count("runs", distinct=True)))
    # Newest first, by deletion or trash time.
    trashed = sorted(trashed, key=lambda e: cast(datetime, e.deleted_at or e.trashed_at), reverse=True)
    return render(request, "ui/panels/site_trash.html", {
        "experiments": trashed,
        "groups": Group.objects.all(),
    })


@require_POST
@site_admin_required
def site_trash_rehome(request, pk):
    """`site_trash_rehome` puts an experiment back into a group, which decides
    who may reach it; the owner stays as is, even when absent.
    """
    experiment = get_object_or_404(policy().trash(request), pk=pk)
    group = get_object_or_404(Group, pk=request.POST.get("group") or 0)
    bin_service.put_back(experiment, group)
    messages.success(request, _("Moved experiment %(pk)s into %(group)s.")
                     % {"pk": experiment.pk, "group": group.name})
    return redirect("ui:site_trash")


@require_POST
@site_admin_required
def site_trash_delete(request, pk):
    """`site_trash_delete` is the only path that destroys a deleted group's
    work."""
    experiment = get_object_or_404(policy().trash(request), pk=pk)
    number = experiment.pk
    delete_experiment_files(experiment)
    messages.success(request, _("Deleted experiment %(pk)s.") % {"pk": number})
    return redirect("ui:site_trash")


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


@site_admin_required
def site_usage(request):
    """`site_usage` totals each group's runs, trials and seconds from `Run`,
    with people listed under their group.
    """
    # Grouping by `Experiment.group` counts each run once, whereas a join
    # through members repeats runs across groups and headcount.
    groups = Group.objects.annotate(
        members=Count("memberships", distinct=True),
        runs=_run_total(Count("pk"), empty=0, experiment__group=OuterRef("pk")),
        trials=_run_total(Sum("trial_count"), experiment__group=OuterRef("pk")),
        seconds=_run_total(Sum("trial_seconds"), experiment__group=OuterRef("pk")),
    )
    return render(request, "ui/panels/site_usage.html", {
        "groups": groups,
        "people": _usage_by_person(),
    })


def _usage_by_person():
    """`_usage_by_person` gives per-account totals within each group and no
    history.
    """
    return (Membership.objects
            .select_related("user", "group")
            # Scoping to the row's group keeps a member's totals from repeating
            # under every group.
            .annotate(**{name: _run_total(agg, empty=empty,
                                          experiment__owner=OuterRef("user"),
                                          experiment__group=OuterRef("group"))
                         for name, agg, empty in (("runs", Count("pk"), 0),
                                                  ("trials", Sum("trial_count"), None),
                                                  ("seconds", Sum("trial_seconds"), None))})
            .order_by("group__name", "user__username"))


@site_admin_required
def site_jobs(request):
    """`site_jobs` lists active runs across every group."""
    active = (Run.objects
              .filter(status__in=("pending", "running"))
              .select_related("experiment", "started_by")
              .order_by("-started_at"))
    return render(request, "ui/panels/site_jobs.html", {
        "runs": [_job_row(run) for run in active],
    })


def _job_row(run) -> dict:
    """What a site admin may see."""
    owner = run.started_by or run.experiment.owner
    # The job belongs to the experiment's group, since one owner may hold
    # several memberships.
    group = run.experiment.group
    return {
        "pk": run.pk,
        "job_id": run.job_id,
        "backend": run.backend,
        "status": run.get_status_display(),
        "started_at": run.started_at,
        "duration": run.duration,
        "trial_count": run.trial_count,
        # A short code that hides what the job is.
        "identifier": run.experiment.identifier,
        "who": owner.get_username() if owner else "",
        "email": getattr(owner, "email", ""),
        "group": group.name if group else "",
    }


@require_POST
@site_admin_required
def site_job_stop(request, pk):
    """`site_job_stop` addresses the run directly, since the site admin lacks
    experiment access for `@experiment_view`. The stop is forcible and also
    cancels a cluster job.
    """
    run = get_object_or_404(Run, pk=pk, status__in=("pending", "running"))
    if run.job_id:
        run_service.abandon_cluster_job(run.job_id)
    run_service.fail_runs([run], _("Stopped by a site administrator."))
    messages.success(request, _("Stopped job %(job)s.")
                     % {"job": run.job_id or run.pk})
    return redirect("ui:site_jobs")
