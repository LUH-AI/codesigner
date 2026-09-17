"""The group and site panels — the two management surfaces.

Their own module rather than more of `ui/views.py`, which is long enough and is
about *experiments*. These are about the people and groups around them, and they
are the two pages that deliberately do not go through `@experiment_view`: a
group lead reaches their colleagues' work through `GroupPolicy`, and a site
admin reaches no experiment at all.

Who may open what is decided here, once, by two decorators. The rules themselves
live in `access/policy.py` — this only asks.
"""

from functools import wraps

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q, Sum
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.models import Group, Membership
from access.policy import is_lead, is_site_admin, membership_of

from .models import Experiment, Run
from .permissions import policy


def lead_required(view):
    """A group lead, and therefore somebody who has a group."""
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        membership = membership_of(request.user)
        if membership is None or not membership.is_lead:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


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


# ── the group panel ──────────────────────────────────────────────────────────

@lead_required
def group_people(request, membership):
    """The group's members, and the seats left to fill.

    A lead fills their group and cannot enlarge it: `user_limit` is a site
    admin's to set, because a lead who could raise their own limit would be a
    lead without one.
    """
    group = membership.group
    return render(request, "ui/panels/group_people.html", {
        "group": group,
        "memberships": group.memberships.select_related("user").all(),
        "seats_left": group.seats_left,
        "roles": Membership.ROLES,
    })


@require_POST
@lead_required
def group_add_person(request, membership):
    """Create an account inside this group.

    A username and a first password, set by the lead and handed over. There is
    still no registration surface anywhere in this and no mail server: an
    account comes into existence because somebody with a group decided it
    should.
    """
    from django.contrib.auth import get_user_model

    group = membership.group
    username = (request.POST.get("username") or "").strip()
    password = request.POST.get("password") or ""
    role = request.POST.get("role") or Membership.MEMBER

    User = get_user_model()
    if not username or not password:
        messages.error(request, _("A username and a first password are needed."))
    elif User.objects.filter(username=username).exists():
        messages.error(request, _("There is already an account called “%(name)s”.")
                       % {"name": username})
    elif group.seats_left < 1:
        # Refused with the number, because "no" without it reads as a bug when
        # the limit is somebody else's to change.
        messages.error(request, _(
            "%(group)s is full: %(limit)s of %(limit)s seats used. A site admin "
            "can raise the limit.") % {"group": group.name,
                                       "limit": group.user_limit})
    elif role not in dict(Membership.ROLES):
        messages.error(request, _("That is not a role."))
    else:
        user = User.objects.create_user(username=username, password=password)
        Membership.objects.create(user=user, group=group, role=role)
        messages.success(request, _("Added %(name)s to %(group)s.")
                         % {"name": username, "group": group.name})
    return redirect("ui:group_people")


@require_POST
@lead_required
def group_remove_person(request, membership, pk):
    """Take somebody out of the group without deleting their account.

    Their experiments stay theirs and become reachable by nobody, which is the
    same state deleting the account would leave — see `access/admin.py`. So this
    refuses while they still own something, and says to reassign first.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    if target.pk == membership.pk:
        messages.error(request, _("You cannot remove yourself from your own group."))
    elif target.user.experiments.exists():
        messages.error(request, _(
            "%(name)s still owns experiments. Out of the group nobody could "
            "reach them — reassign them first.") % {"name": target.user.get_username()})
    else:
        name = target.user.get_username()
        target.delete()
        messages.success(request, _("Removed %(name)s from the group.") % {"name": name})
    return redirect("ui:group_people")


@lead_required
def group_work(request, membership):
    """Everyone else's experiments in this group.

    Separate from the lead's own Experiments tab on purpose: being a lead should
    not quietly turn the everyday list into everyone's. `GroupPolicy` draws the
    same split — `for_listing` against `group_experiments`.
    """
    return render(request, "ui/panels/group_work.html", {
        "group": membership.group,
        "experiments": policy().group_experiments(request)
                              .select_related("owner").order_by("-created_at"),
    })


# ── the site panel ───────────────────────────────────────────────────────────

@site_admin_required
def site_groups(request):
    """Every group, its size and whether it is active.

    No experiments anywhere on this page or the two below it. A site admin
    manages groups; showing them colleagues' work is the thing the separation
    exists to prevent.
    """
    return render(request, "ui/panels/site_groups.html", {
        "groups": Group.objects.annotate(members=Count("memberships")),
    })


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

    if not name or limit < 0:
        messages.error(request, _("A group needs a name and a seat limit of zero or more."))
        return redirect("ui:site_groups")

    if pk:
        group = get_object_or_404(Group, pk=pk)
        group.name = name
        group.user_limit = limit
        group.is_active = bool(request.POST.get("is_active"))
        group.save(update_fields=["name", "user_limit", "is_active"])
        messages.success(request, _("Saved %(name)s.") % {"name": group.name})
    elif Group.objects.filter(name=name).exists():
        messages.error(request, _("There is already a group called “%(name)s”.")
                       % {"name": name})
    else:
        Group.objects.create(name=name, user_limit=limit)
        messages.success(request, _("Created %(name)s.") % {"name": name})
    return redirect("ui:site_groups")


@site_admin_required
def site_usage(request):
    """What each group is using — and nothing about what they are using it for.

    Aggregated from `Run`, which carries `trial_count` and `trial_seconds`
    already, so this counts what was actually spent rather than estimating it.
    Group first, with the people underneath, because the unit being managed is
    the group.
    """
    groups = Group.objects.annotate(
        members=Count("memberships", distinct=True),
        runs=Count("memberships__user__experiments__runs", distinct=True),
        trials=Sum("memberships__user__experiments__runs__trial_count"),
        seconds=Sum("memberships__user__experiments__runs__trial_seconds"),
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
            .annotate(runs=Count("user__experiments__runs", distinct=True),
                      trials=Sum("user__experiments__runs__trial_count"),
                      seconds=Sum("user__experiments__runs__trial_seconds"))
            .order_by("group__name", "user__username"))


#: What a site admin is shown about a run. Names the fields rather than the
#: model so that adding one to `Run` does not quietly widen this: experiment
#: names, models, metrics and scores are all deliberately absent, and the only
#: thing identifying the work is the experiment's own opaque identifier, which
#: is what makes a job nameable in a support conversation.
JOB_FIELDS = ("id", "job_id", "status", "started_at", "finished_at",
              "trial_count", "trial_seconds")


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
    membership = membership_of(owner) if owner else None
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
        "group": membership.group.name if membership else "",
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
    from django.utils import timezone

    from .views import _abandon_cluster_job

    run = get_object_or_404(Run, pk=pk, status__in=("pending", "running"))
    if run.job_id:
        _abandon_cluster_job(run.job_id)
    Run.objects.filter(pk=run.pk).update(
        status="error", finished_at=timezone.now(),
        error=_("Stopped by a site administrator."))
    messages.success(request, _("Stopped job %(job)s.")
                     % {"job": run.job_id or run.pk})
    return redirect("ui:site_jobs")
