"""Each `.ServiceError` raised in `groups` carries the reason its operation
cannot proceed.
"""

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models.deletion import ProtectedError
from django.utils.translation import gettext as _

from access.models import Group, Membership
from access.onboarding import onboarding
from access.services.errors import ServiceError


def _join(group, identifier: str, role: str):
    hook = onboarding()
    try:
        user = hook.find(identifier)
        created = user is None
        if created:
            user = hook.create(identifier)
        Membership.objects.create(user=user, group=group, role=role)
    except ValidationError:
        raise ServiceError(_("“%(who)s” is not a valid %(what)s.")
                           % {"who": identifier, "what": str(hook.label).lower()})
    return user, created


def create_group(name: str, seats: int, lead: str):
    """`create_group` creates the group and its primary lead *lead* in one
    transaction, since a leadless group could never appoint leads. It returns
    `(group, user, message)`.
    """
    name = (name or "").strip()
    identifier = onboarding().normalize(lead)
    if not name:
        raise ServiceError(_("A group needs a name."))
    if Group.objects.filter(name=name).exists():
        raise ServiceError(_("There is already a group called “%(name)s”.") % {"name": name})
    if not identifier:
        raise ServiceError(_("A new group needs a primary lead."))
    if seats < 1:
        raise ServiceError(_("A new group needs at least one seat, for its primary lead."))
    with transaction.atomic():
        group = Group.objects.create(name=name, user_limit=seats)
        user, created = _join(group, identifier, Membership.PRIMARY_LEAD)
    return group, user, onboarding().welcome(user, created, group)


def update_group(group, *, name: str, seats: int, active: bool,
                 storage_gb: float | None = None) -> None:
    from experiments.services.storage import forget

    name = (name or "").strip()
    if not name or seats < 0 or (storage_gb is not None and storage_gb < 0):
        raise ServiceError(_("A group needs a name, a seat limit of zero or more, and a "
                             "storage limit of zero or more."))
    if Group.objects.filter(name=name).exclude(pk=group.pk).exists():
        raise ServiceError(_("There is already a group called “%(name)s”.") % {"name": name})
    group.name, group.user_limit, group.is_active = name, seats, active
    if storage_gb is not None:
        group.storage_limit_bytes = int(storage_gb * 1024 ** 3)
    group.save(update_fields=["name", "user_limit", "is_active", "storage_limit_bytes"])
    forget(group)


def delete_group(group) -> None:
    """`delete_group` raises `.ServiceError` while the group has members, since
    `.Membership.group` is `PROTECT`."""
    try:
        group.delete()
    except ProtectedError:
        raise ServiceError(_("%(name)s still has people in it. Remove them first.")
                           % {"name": group.name})


def grantable_roles(membership) -> list[tuple[str, str]]:
    """`grantable_roles` only offers the lead role to a primary lead."""
    if membership.is_primary_lead:
        return [(Membership.MEMBER, "Member"), (Membership.LEAD, "Group lead")]
    return [(Membership.MEMBER, "Member")]


def add_person(group, identifier: str, role: str = Membership.MEMBER, *, allowed_roles=None):
    """`add_person` adds *identifier* to *group*, creating the account when
    missing, and returns `(user, created, message)`. Without *allowed_roles*,
    `add_person` also accepts the primary role for a group that has none;
    otherwise the role moves by `transfer_primary`.
    """
    identifier = onboarding().normalize(identifier)
    allowed = set(allowed_roles) if allowed_roles is not None else {
        Membership.MEMBER, Membership.LEAD}
    if allowed_roles is None and not group.memberships.filter(
            role=Membership.PRIMARY_LEAD).exists():
        allowed.add(Membership.PRIMARY_LEAD)
    if not identifier:
        raise ServiceError(_("Say who to add."))
    if group.seats_left < 1:
        raise ServiceError(_(
            "%(group)s is full: %(limit)s of %(limit)s seats used. A site admin "
            "can raise the limit.") % {"group": group.name, "limit": group.user_limit})
    if role not in allowed:
        raise ServiceError(_("You cannot add somebody with that role."))
    existing = onboarding().find(identifier)
    if existing is not None and group.memberships.filter(user=existing).exists():
        raise ServiceError(_("%(who)s is already in %(group)s.")
                           % {"who": identifier, "group": group.name})
    with transaction.atomic():
        user, created = _join(group, identifier, role)
    return user, created, onboarding().welcome(user, created, group)


def remove_person(target, *, by=None) -> None:
    """`remove_person` raises `.ServiceError` while the person owns experiments
    in the group, since nobody could reach those afterwards. The account
    survives."""
    name = target.user.get_username()
    if by is not None and target.pk == by.pk:
        raise ServiceError(_("You cannot remove yourself from your own group."))
    if target.is_primary_lead:
        raise ServiceError(_(
            "%(name)s is the primary lead. The role has to be transferred "
            "before they can leave.") % {"name": name})
    if target.user.experiments.filter(group_id=target.group_id).exists():
        raise ServiceError(_(
            "%(name)s still owns experiments. Out of the group nobody could "
            "reach them — reassign them first.") % {"name": name})
    target.delete()


def set_role(target, role: str, *, by=None) -> None:
    if by is not None and target.pk == by.pk:
        raise ServiceError(_(
            "You cannot change your own role. Transfer the primary lead role "
            "to somebody else instead."))
    if role not in (Membership.MEMBER, Membership.LEAD):
        raise ServiceError(_("That is not a role you can set."))
    target.role = role
    target.save(update_fields=["role"])


def transfer_primary(current, target) -> None:
    """`transfer_primary` demotes *current* before promoting *target*, since
    `one_primary_lead_per_group` allows one primary lead per group."""
    if target.pk == current.pk:
        raise ServiceError(_("You already hold it."))
    if target.role != Membership.LEAD:
        raise ServiceError(_(
            "The primary lead role can only go to a group lead. Make "
            "%(name)s a lead first.") % {"name": target.user.get_username()})
    with transaction.atomic():
        current.role = Membership.LEAD
        current.save(update_fields=["role"])
        target.role = Membership.PRIMARY_LEAD
        target.save(update_fields=["role"])
