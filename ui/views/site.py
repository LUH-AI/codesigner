"""The site admin's pages: groups, the trash, usage and running jobs.

A site admin reaches no experiment at all, so none of these go through
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
from django.utils import timezone
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
    """Holds `manage_site`. Deliberately not "is a superuser": see
    `access/management/commands/create_site_admin.py` on why the role is a
    permission rather than a rank."""
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not is_site_admin(request.user):
            raise Http404
        return view(request, *args, **kwargs)
    return wrapped


@site_admin_required
def site_groups(request):
    """Every group, its size and whether it is active.

    No experiments anywhere on this page or the two below it. A site admin
    manages groups; showing them colleagues' work is the thing the separation
    exists to prevent.
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
    """Create a group, or change one's limit or whether it is active."""
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
        if storage_gb == storage_gb:  # not NaN: a limit was given
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
    """A new group, and the primary lead it cannot be created without.

    A group with nobody able to appoint a lead is a group that cannot grow —
    the reason `access/migrations/0007` exists — so the two are made together
    or not at all. The lead is named by email, and an unknown address creates
    the account, the same door `group_add_person` uses.

    A seat limit of zero is refused here for the same reason: the primary lead
    occupies a seat, and a group created already over its own limit is a state
    nothing else in this module expects.
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
    """Remove a group, setting its work aside rather than destroying it.

    `Group.delete()` empties the group's experiments into the bin first — see
    `access/models.py` on why that rather than taking them with it.

    A group that still has people in it is refused. `Membership.group` is
    PROTECT and always has been: emptying a group of its accounts is a separate
    decision, made person by person on the group panel, and doing it as a side
    effect of one button would delete accounts nobody asked to delete.
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
    """The site's bin, and the two things to do about what is in it.

    Two ways in: a deleted group's work, and what somebody deleted that none of
    its people kept — see `experiments/services/bin.py`. Settled the same way either way,
    so listed together, each saying which it is.

    Shown with the same discipline as the jobs panel: an opaque identifier, who
    used to own it and how much it cost, and nothing about what it was *for*.
    Deciding whether work should be rehomed or discarded does not require
    reading it, and a bin that listed experiment names would be a way around the
    boundary rather than a consequence of one.
    """
    trashed = (policy().trash(request)
               .select_related("owner")
               .annotate(run_count=Count("runs", distinct=True)))
    # Newest first, by whichever of the two put it here.
    trashed = sorted(trashed, key=lambda e: cast(datetime, e.deleted_at or e.trashed_at), reverse=True)
    return render(request, "ui/panels/site_trash.html", {
        "experiments": trashed,
        "groups": Group.objects.all(),
    })


@require_POST
@site_admin_required
def site_trash_rehome(request, pk):
    """Put an experiment from the site's bin into a group again.

    Which makes it that group's — its lead can reach it, and whoever it was
    shared with can again. The owner is left as it was, including when that is
    nobody: an experiment can outlive the account that made it, and a group is
    what decides who may see it. See `bin.put_back`.
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
    """Discard one for good. The only route by which a group's deletion
    eventually destroys anything, and it is a second, separate decision."""
    experiment = get_object_or_404(policy().trash(request), pk=pk)
    number = experiment.pk
    delete_experiment_files(experiment)
    messages.success(request, _("Deleted experiment %(pk)s.") % {"pk": number})
    return redirect("ui:site_trash")


def _run_total(aggregate, *, empty=None, **scope):
    """One aggregate over the runs matching *scope*, as a correlated subquery.

    A subquery rather than a join so that it cannot be multiplied by whatever
    else the outer query joins: beside `Count("memberships")`, a joined sum
    counts each run once per member.

    Grouped by exactly the fields *scope* correlates on, so the inner query
    yields one row per outer row. With no runs it yields none, which reads as
    null — *empty* is what to say instead, so a count keeps saying 0 as the
    joined `Count` it replaces did, while a sum keeps saying nothing.
    """
    runs = (Run.objects.filter(**scope).order_by()
            .values(*scope)
            .annotate(value=aggregate)
            .values("value")[:1])
    total = Subquery(runs)
    return Coalesce(total, empty) if empty is not None else total


@site_admin_required
def site_usage(request):
    """What each group is using — and nothing about what they are using it for.

    Aggregated from `Run`, which carries `trial_count` and `trial_seconds`
    already, so this counts what was actually spent rather than estimating it.
    Group first, with the people underneath, because the unit being managed is
    the group.
    """
    # The group's own work, through `Experiment.group`, and summed in a
    # subquery. Two things were wrong with the join through its members: it
    # counted a person's runs under every group they are in, and a `Sum`
    # joined alongside `Count("memberships")` is multiplied by the headcount.
    # Each run belongs to exactly one group, so each is counted exactly once.
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
    """The same totals per account, for the group each belongs to.

    Deliberately totals and not a history: how much somebody has spent is a
    thing a site admin may need to know; what they spent it on is not.
    """
    return (Membership.objects
            .select_related("user", "group")
            # What they spent *in this group*. The same person appears once per
            # group they are in, and each row has to be about that group's work
            # or their totals repeat in full under every one of them.
            .annotate(**{name: _run_total(agg, empty=empty,
                                          experiment__owner=OuterRef("user"),
                                          experiment__group=OuterRef("group"))
                         for name, agg, empty in (("runs", Count("pk"), 0),
                                                  ("trials", Sum("trial_count"), None),
                                                  ("seconds", Sum("trial_seconds"), None))})
            .order_by("group__name", "user__username"))


@site_admin_required
def site_jobs(request):
    """What is running now, across every group, with a way to stop it."""
    active = (Run.objects
              .filter(status__in=("pending", "running"))
              .select_related("experiment", "started_by")
              .order_by("-started_at"))
    return render(request, "ui/panels/site_jobs.html", {
        "runs": [_job_row(run) for run in active],
    })


def _job_row(run) -> dict:
    """One run, projected to what a site admin may see."""
    owner = run.started_by or run.experiment.owner
    # The experiment's group, which is the recorded boundary. It used to be read
    # off the owner's membership, and a person may now be in several — the job
    # belongs to the group its work does, whichever of theirs that is.
    group = run.experiment.group
    return {
        "pk": run.pk,
        "job_id": run.job_id,
        "backend": run.backend,
        "status": run.get_status_display(),
        "started_at": run.started_at,
        "duration": run.duration,
        "trial_count": run.trial_count,
        # The experiment's own short code: enough to talk about a job, and it
        # says nothing about what the job is.
        "identifier": run.experiment.identifier,
        "who": owner.get_username() if owner else "",
        "email": getattr(owner, "email", ""),
        "group": group.name if group else "",
    }


@require_POST
@site_admin_required
def site_job_stop(request, pk):
    """Stop a run from outside its group.

    Addressed by **run**, not by experiment: a site admin has no experiment
    visibility for `@experiment_view` to check, and giving them some so that a
    decorator would fit would defeat the separation. This is a second, narrower
    door into the same action, and `tests/ui/test_permissions.py` knows it is
    one.

    Forcible by nature. There is no cooperative version from here — a site admin
    stopping somebody else's job is already the escalation.
    """
    run = get_object_or_404(Run, pk=pk, status__in=("pending", "running"))
    if run.job_id:
        run_service.abandon_cluster_job(run.job_id)
    Run.objects.filter(pk=run.pk).update(
        status="error", finished_at=timezone.now(),
        error=_("Stopped by a site administrator."))
    messages.success(request, _("Stopped job %(job)s.")
                     % {"job": run.job_id or run.pk})
    return redirect("ui:site_jobs")
