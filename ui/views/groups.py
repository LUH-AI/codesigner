"""The group views check membership with the decorators below instead of
`@experiment_view`.
"""

from functools import wraps

from django.contrib import messages
from django.db.models import Q, Count, OuterRef, Prefetch, Subquery
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.models import Membership
from access.policy import membership_in, memberships_of, stronger
from access.onboarding import onboarding
from access.services import groups as group_service
from access.services.errors import ServiceError
from experiments.models import Experiment, ExperimentShare
from experiments.services.storage import usage as storage_usage

from ..formatting import human


def member_required(view):
    """`member_required` admits any member of the URL's group.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


def lead_required(view):
    """`lead_required` passes the view a lead's membership of the URL's group. A
    non-member and a non-lead member get the same 404, so the answer hides which
    groups exist.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None or not membership.is_lead:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


def primary_lead_required(view):
    """`primary_lead_required` only admits the primary lead, who appoints the
    other leads.
    """
    @wraps(view)
    def wrapped(request, group_pk, *args, **kwargs):
        membership = membership_in(request.user, group_pk)
        if membership is None or not membership.is_primary_lead:
            raise Http404
        return view(request, membership, *args, **kwargs)
    return wrapped


def group_index(request):
    """`group_index` opens the person's first group by name, or a page saying
    there is none; a remembered choice would make tabs fight over one session
    slot.
    """
    groups = sorted((m.group for m in memberships_of(request.user)),
                    key=lambda g: g.name.lower())
    if groups:
        return redirect("ui:group_detail", group_pk=groups[0].pk)
    return render(request, "ui/panels/group_none.html")


@member_required
def group_detail(request, membership):
    """`group_detail` shows every member the roster, their own experiments with
    sharing controls, and what colleagues shared. A lead also sees the rest of
    the group's experiments and the people controls.
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
    shared = list(live.exclude(owner=user)
                  .filter(Q(shares__user=user) | ~Q(group_share=""))
                  .annotate(own=Subquery(level.values("level")[:1])).distinct())
    for exp in shared:
        exp.level = stronger(exp.own, exp.group_share)
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
        # Primary moves only by transfer, never by grant.
        "roles": group_service.grantable_roles(membership) if membership.is_lead else [],
        "onboarding": onboarding(),
        "is_lead": membership.is_lead,
        "is_primary": membership.is_primary_lead,
        "mine": mine,
        "shared": shared,
        "others": others,
        "levels": ExperimentShare.LEVELS,
        "here": reverse("ui:group_detail", args=[group.pk]),
    })


@require_POST
@lead_required
def group_add_person(request, membership):
    """`group_add_person` adds a person through `.onboarding`, which decides
    how a new account gets its password. The new member takes a seat."""
    try:
        _user, _created, message = group_service.add_person(
            membership.group, request.POST.get("identifier") or "",
            request.POST.get("role") or Membership.MEMBER,
            allowed_roles=dict(group_service.grantable_roles(membership)))
    except ServiceError as error:
        messages.error(request, str(error))
    else:
        messages.success(request, message)
    return redirect("ui:group_detail", group_pk=membership.group_id)


@require_POST
@lead_required
def group_remove_person(request, membership, pk):
    """`group_remove_person` removes a person only once that person owns no
    experiments here, since nobody could reach those experiments afterwards. The
    account survives.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    name = target.user.get_username()
    try:
        group_service.remove_person(target, by=membership)
    except ServiceError as error:
        messages.error(request, str(error))
    else:
        messages.success(request, _("Removed %(name)s from the group.") % {"name": name})
    return redirect("ui:group_detail", group_pk=membership.group_id)


@require_POST
@primary_lead_required
def group_set_role(request, membership, pk):
    """`group_set_role` switches a member between lead and member; the primary
    role only moves by transfer.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    try:
        group_service.set_role(target, request.POST.get("role") or "", by=membership)
    except ServiceError as error:
        messages.error(request, str(error))
    else:
        messages.success(request, _("%(name)s is now a %(role)s.") % {
            "name": target.user.get_username(),
            "role": target.get_role_display().lower()})
    return redirect("ui:group_detail", group_pk=membership.group_id)


@require_POST
@primary_lead_required
def group_transfer_primary(request, membership, pk):
    """`group_transfer_primary` hands the role only to an existing lead. Both
    rows change in one transaction, since the constraint allows one primary lead
    per group.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    try:
        group_service.transfer_primary(membership, target)
    except ServiceError as error:
        messages.error(request, str(error))
    else:
        messages.success(request, _(
            "%(name)s is now the primary lead of %(group)s. You are a group "
            "lead.") % {"name": target.user.get_username(),
                        "group": membership.group.name})
    return redirect("ui:group_detail", group_pk=membership.group_id)


def _storage_line(group):
    """`_storage_line` formats the group's storage figures."""
    now = storage_usage(group)
    return {"used": human(now["total"]), "limit": human(now["limit"]),
            "models": human(now["models"]), "percent": min(100, round(100 * now["share"]))}
