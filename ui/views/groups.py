"""The group views check membership with the decorators below instead of
`@experiment_view`.
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
    """`primary_lead_required` admits only the primary lead, who appoints the
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
        # Primary moves only by transfer, never by grant.
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
    """Only a primary lead may grant the lead role.
    """
    if membership.is_primary_lead:
        return [(Membership.MEMBER, "Member"), (Membership.LEAD, "Group lead")]
    return [(Membership.MEMBER, "Member")]


@require_POST
@lead_required
def group_add_person(request, membership):
    """`group_add_person` joins an existing account by email or creates one with
    a first password the lead hands over. Either way the new member takes a
    seat.
    """

    group = membership.group
    address = normalize(request.POST.get("email"))
    password = request.POST.get("password") or ""
    role = request.POST.get("role") or Membership.MEMBER

    if not address:
        messages.error(request, _("An email address is needed."))
    elif group.seats_left < 1:
        messages.error(request, _(
            "%(group)s is full: %(limit)s of %(limit)s seats used. A site admin "
            "can raise the limit.") % {"group": group.name,
                                       "limit": group.user_limit})
    elif role not in dict(_grantable_roles(membership)):
        # The server checks the role because posts can bypass the form.
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
    """`group_remove_person` refuses while the person still owns experiments
    here, since nobody could reach them afterwards. The account survives.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    if target.pk == membership.pk:
        messages.error(request, _("You cannot remove yourself from your own group."))
    elif target.is_primary_lead:
        # Only a second lead can try to remove the primary lead, whose departure
        # would leave nobody to appoint leads.
        messages.error(request, _(
            "%(name)s is the primary lead. The role has to be transferred "
            "before they can leave.") % {"name": target.user.get_username()})
    # Only this group's experiments count.
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
    """`group_set_role` switches a member between lead and member; the primary
    role moves only by transfer.
    """
    target = get_object_or_404(Membership, pk=pk, group=membership.group)
    role = request.POST.get("role") or ""

    if target.pk == membership.pk:
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
    """`group_transfer_primary` hands the role only to an existing lead. Both
    rows change in one transaction, since the constraint allows one primary lead
    per group.
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
    """`_storage_line` formats the group's storage figures."""
    now = storage_usage(group)
    return {"used": human(now["total"]), "limit": human(now["limit"]),
            "models": human(now["models"]), "percent": min(100, round(100 * now["share"]))}
