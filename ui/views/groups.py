"""A group's own page: its people, their roles, and its storage.

Like the site's pages (`site.py`), it deliberately does not go through
`@experiment_view`: a group lead reaches their colleagues' work through
`GroupPolicy`. Who may open what is decided here, once, by decorators. The
rules themselves live in `access/policy.py` — this only asks.
"""

from functools import wraps

from django.contrib import messages
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q, Count, OuterRef, Prefetch, Subquery
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.models import Membership
from access.policy import membership_in, memberships_of
from access.services.accounts import find_or_create_by_email, normalize
from experiments.models import Experiment, ExperimentShare
from experiments.services.storage import usage as storage_usage

from ..formatting import human


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
        "storage": _storage_line(group),
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


def _storage_line(group):
    """What the group page says of the group's storage."""
    now = storage_usage(group)
    return {"used": human(now["total"]), "limit": human(now["limit"]),
            "models": human(now["models"]), "percent": min(100, round(100 * now["share"]))}
