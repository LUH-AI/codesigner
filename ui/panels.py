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
from django.db import transaction
from django.db.models import Count, OuterRef, Q, Subquery, Sum
from django.db.models.functions import Coalesce
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from django.db.models.deletion import ProtectedError
from django.utils import timezone

from access.models import Group, Membership
from access.policy import is_site_admin, membership_in, memberships_of

from .models import Experiment, Run
from .permissions import policy


def member_required(view):
    """Somebody in the group this URL names, in whatever role.

    The group page is everybody's — who is in it, and what they can reach in
    it. What a lead sees in addition is decided on the page, not by a second
    route.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


def lead_required(view):
    """A lead of the group this URL names.

    The membership handed to the view is *that group's* — which is what every
    body below already assumes. It used to be "their membership", singular; the
    URL says which one now, because a person may hold several and none of them
    is the obvious default.

    A 404 for a group you are in but do not lead, and the same 404 for one you
    are not in at all. A page that is not yours is a page that is not there,
    and a different answer for the two would report which groups exist.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None or not membership.is_lead:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


def primary_lead_required(view):
    """The one lead of the group this URL names who may appoint the others.

    A separate decorator rather than a check inside `lead_required`, because the
    difference is the whole point of the role: everything a lead does stays open
    to every lead, and only these views narrow to one person.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None or not membership.is_primary_lead:
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

def group_index(request):
    """Where the Group tab goes: the first of this person's groups, by name.

    The rail links here rather than to a group, so that it never has to compute
    a per-person URL, and the question "which group" is answered once, here.
    First by name, not "the last one you looked at": held in the session, that
    would be two tabs fighting over one slot.

    Somebody in no group gets a page saying so rather than a missing tab — the
    tab is everybody's, and "you are not in one yet" is an answer worth giving.
    """
    groups = sorted((m.group for m in memberships_of(request.user)),
                    key=lambda g: g.name.lower())
    if groups:
        return redirect("ui:group_detail", group_pk=groups[0].pk)
    return render(request, "ui/panels/group_none.html")


@member_required
def group_detail(request, membership):
    """One group: its people, and its experiments as this person may reach them.

    Everybody in the group sees who else is in it. Their own experiments here
    come with the controls for sharing each one, since sharing is how work
    moves within a group; what colleagues shared with them is listed with the
    level they were given. A lead sees two more things — everybody else's
    experiments in the group, and the controls for its people — because
    reaching those is what the role is for.
    """
    from django.db.models import Prefetch

    from .models import ExperimentShare

    group, user = membership.group, request.user
    live = (Experiment.objects.select_related("data", "owner")
            .filter(group=group, trashed_at__isnull=True, deleted_at__isnull=True)
            .order_by("-created_at"))
    memberships = list(
        group.memberships.select_related("user")
        .annotate(owned=Count("user__experiments", distinct=True, filter=Q(
            user__experiments__group=group,
            user__experiments__trashed_at__isnull=True,
            user__experiments__deleted_at__isnull=True)))
        .order_by("user__username"))
    colleagues = [m.user for m in memberships if m.user_id != user.pk]

    mine = []
    for exp in live.filter(owner=user).prefetch_related(
            Prefetch("shares", queryset=ExperimentShare.objects.select_related("user"))):
        levels = {share.user_id: share.level for share in exp.shares.all()}
        mine.append({"experiment": exp,
                     "shares": [s for s in exp.shares.all()],
                     "colleagues": [{"user": c, "level": levels.get(c.pk, "none")}
                                    for c in colleagues]})

    level = ExperimentShare.objects.filter(experiment=OuterRef("pk"), user=user)
    shared = list(live.filter(shares__user=user)
                  .annotate(level=Subquery(level.values("level")[:1])))
    others = []
    if membership.is_lead:
        others = list(live.exclude(owner=user).exclude(pk__in=[e.pk for e in shared])
                      .annotate(people=Count("shares", distinct=True)))

    return render(request, "ui/panels/group.html", {
        "group": group,
        "membership": membership,
        "memberships": memberships,
        "seats_left": group.seats_left,
        # What this person may hand out, which is not the whole list. Primary
        # is never here: it is transferred from whoever holds it, never granted
        # alongside a new account, or a group would end up with two.
        "roles": _grantable_roles(membership) if membership.is_lead else [],
        "is_lead": membership.is_lead,
        "is_primary": membership.is_primary_lead,
        "mine": mine,
        "shared": shared,
        "others": others,
        "levels": ExperimentShare.LEVELS,
        "here": reverse("ui:group_detail", args=[group.pk]),
    })


def _grantable_roles(membership):
    """The roles *this* lead may give somebody.

    A primary lead may create leads; an ordinary lead may not, or the one
    restriction the role carries would be one form post away from meaningless.
    """
    if membership.is_primary_lead:
        return [(Membership.MEMBER, "Member"), (Membership.LEAD, "Group lead")]
    return [(Membership.MEMBER, "Member")]


@require_POST
@lead_required
def group_add_person(request, membership):
    """Put somebody in this group, by their email address.

    An address somebody here already has just joins them to the group: nobody
    gets a second account, and nothing about the one they have changes. An
    unknown address creates the account as well, with a first password the lead
    hands over — there is still no registration surface and no mail server, so
    an account comes into existence because somebody with a group decided it
    should.

    The seat limit applies either way. An existing account joining still takes
    a seat, or the limit would be one email address away from meaningless.
    """
    from django.core.exceptions import ValidationError

    from access.services.accounts import find_or_create_by_email, normalize

    group = membership.group
    address = normalize(request.POST.get("email"))
    password = request.POST.get("password") or ""
    role = request.POST.get("role") or Membership.MEMBER

    if not address:
        messages.error(request, _("An email address is needed."))
    elif group.seats_left < 1:
        # Refused with the number, because "no" without it reads as a bug when
        # the limit is somebody else's to change.
        messages.error(request, _(
            "%(group)s is full: %(limit)s of %(limit)s seats used. A site admin "
            "can raise the limit.") % {"group": group.name,
                                       "limit": group.user_limit})
    elif role not in dict(_grantable_roles(membership)):
        # Covers both a role that does not exist and one this lead may not
        # give. Checked here and not only in the form, because the form is not
        # what arrives.
        messages.error(request, _("You cannot add somebody with that role."))
    elif group.memberships.filter(user__email__iexact=address).exists():
        messages.error(request, _("%(email)s is already in %(group)s.")
                       % {"email": address, "group": group.name})
    else:
        try:
            with transaction.atomic():
                user, created, given = find_or_create_by_email(address, password=password)
                Membership.objects.create(user=user, group=group, role=role)
        except ValidationError:
            messages.error(request, _("“%(email)s” is not an email address.")
                           % {"email": address})
        else:
            if created:
                # The one time the password exists in readable form, said once
                # — a message is shown on the next page and then gone.
                messages.success(request, _(
                    "Created an account for %(email)s and added it to %(group)s. "
                    "Its first password is %(password)s — hand it over; they can "
                    "change it under Settings → Account.") % {
                        "email": address, "group": group.name, "password": given})
            else:
                messages.success(request, _("Added %(email)s to %(group)s.")
                                 % {"email": address, "group": group.name})
    return redirect("ui:group_detail", group_pk=membership.group_id)


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
    elif target.is_primary_lead:
        # Only reachable by a second lead trying it, since the primary lead
        # cannot remove themselves by the branch above. Either way the group
        # would be left with nobody able to appoint a lead.
        messages.error(request, _(
            "%(name)s is the primary lead. The role has to be transferred "
            "before they can leave.") % {"name": target.user.get_username()})
    # Their work in *this* group. They may own experiments in another, and
    # those stay reachable through that group whatever happens here.
    elif target.user.experiments.filter(group_id=membership.group_id).exists():
        messages.error(request, _(
            "%(name)s still owns experiments. Out of the group nobody could "
            "reach them — reassign them first.") % {"name": target.user.get_username()})
    else:
        name = target.user.get_username()
        target.delete()
        messages.success(request, _("Removed %(name)s from the group.") % {"name": name})
    return redirect("ui:group_detail", group_pk=membership.group_id)


@require_POST
@primary_lead_required
def group_set_role(request, membership, pk):
    """Make somebody a lead, or make them an ordinary member again.

    The one thing a primary lead can do that a lead cannot. Primary is not
    among the roles this will set — that one moves by transfer, from the person
    who holds it, so that a group can never be left with two or with none.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    role = request.POST.get("role") or ""

    if target.pk == membership.pk:
        # Standing down is `group_transfer_primary`, which hands the role to a
        # named successor. Doing it here would leave the group with no primary
        # lead and nobody able to appoint one.
        messages.error(request, _(
            "You cannot change your own role. Transfer the primary lead role "
            "to somebody else instead."))
    elif role not in (Membership.MEMBER, Membership.LEAD):
        messages.error(request, _("That is not a role you can set."))
    else:
        target.role = role
        target.save(update_fields=["role"])
        messages.success(request, _("%(name)s is now a %(role)s.") % {
            "name": target.user.get_username(),
            "role": target.get_role_display().lower()})
    return redirect("ui:group_detail", group_pk=membership.group_id)


@require_POST
@primary_lead_required
def group_transfer_primary(request, membership, pk):
    """Hand the primary lead role to another lead in this group.

    To a lead rather than to anybody: the role is a lead's with one addition,
    so handing it to a member would promote and appoint in a single unexplained
    step. Make them a lead first, then hand it over — two decisions, which is
    what it is.

    Both rows are written in one transaction. The constraint allows one primary
    lead per group, so the order matters and a half-applied transfer would
    either leave two or leave none.
    """
    from django.db import transaction

    target = get_object_or_404(Membership, pk=pk, group=membership.group)

    if target.pk == membership.pk:
        messages.error(request, _("You already hold it."))
    elif target.role != Membership.LEAD:
        messages.error(request, _(
            "The primary lead role can only go to a group lead. Make "
            "%(name)s a lead first.") % {"name": target.user.get_username()})
    else:
        with transaction.atomic():
            membership.role = Membership.LEAD
            membership.save(update_fields=["role"])
            target.role = Membership.PRIMARY_LEAD
            target.save(update_fields=["role"])
        messages.success(request, _(
            "%(name)s is now the primary lead of %(group)s. You are a group "
            "lead.") % {"name": target.user.get_username(),
                        "group": membership.group.name})
    return redirect("ui:group_detail", group_pk=membership.group_id)


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
    from django.core.exceptions import ValidationError

    from access.services.accounts import find_or_create_by_email, normalize

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
    """What group deletions left behind, and the two things to do about it.

    Shown with the same discipline as the jobs panel: an opaque identifier, who
    used to own it and how much it cost, and nothing about what it was *for*.
    Deciding whether work should be rehomed or discarded does not require
    reading it, and a bin that listed experiment names would be a way around the
    boundary rather than a consequence of one.
    """
    trashed = (policy().trash(request)
               .select_related("owner")
               .annotate(run_count=Count("runs", distinct=True))
               .order_by("-trashed_at"))
    return render(request, "ui/panels/site_trash.html", {
        "experiments": trashed,
        "groups": Group.objects.all(),
    })


@require_POST
@site_admin_required
def site_trash_rehome(request, pk):
    """Put a trashed experiment into a group again.

    Which makes it that group's — its lead can reach it, and its members can if
    the owner had shared it. The owner is left as it was, including when that is
    nobody: an experiment can outlive the account that made it, and a group is
    what decides who may see it.
    """
    experiment = get_object_or_404(policy().trash(request), pk=pk)
    group = get_object_or_404(Group, pk=request.POST.get("group") or 0)
    experiment.group = group
    experiment.trashed_at = None
    experiment.save(update_fields=["group", "trashed_at"])
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
    experiment.delete()
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
