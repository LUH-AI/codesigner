"""A function in `users` changes nothing when it raises `.ServiceError`, whose
message gives the reason.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.db.models import Case, CharField, Count, Exists, OuterRef, Value, When
from django.utils import timezone
from django.utils.translation import gettext as _

from access.backends import account_named
from access.models import DISPLAY_NAME_MIN, account_of, shown_name
from access.services.errors import ServiceError


def named(who: str):
    """`named` raises `.ServiceError` when no account matches *who*."""
    user = account_named(who)
    if user is None:
        raise ServiceError(_("There is no account for %(who)s.") % {"who": who})
    return user


#: Each account has one status, worked out from its active flag, its pending
#: password and its setup links.
ACTIVE = "active"
INVITED = "invited"
INVITE_EXPIRED = "invite expired"
DEACTIVATED = "deactivated"
STATUSES = (ACTIVE, INVITED, INVITE_EXPIRED, DEACTIVATED)


def with_status(users):
    """`with_status` annotates each of *users* with `status`: deactivated
    when switched off, invited while a password is still to be set (by a live
    `.SetupLink`, or at the first sign-in), invite expired once every link
    lapsed, and active otherwise."""
    from access.models import Account, SetupLink

    live_link = SetupLink.objects.filter(user=OuterRef("pk"), used_at__isnull=True,
                                         expires_at__gt=timezone.now())
    return users.annotate(status=Case(
        When(is_active=False, then=Value(DEACTIVATED)),
        When(account__password_pending=Account.SETUP_LINK, then=Case(
            When(Exists(live_link), then=Value(INVITED)), default=Value(INVITE_EXPIRED))),
        When(account__password_pending=Account.FIRST_SIGN_IN, then=Value(INVITED)),
        default=Value(ACTIVE), output_field=CharField()))


def list_users():
    return (with_status(get_user_model().objects)
            .annotate(experiment_count=Count("experiments"))
            .prefetch_related("memberships__group")
            .order_by("username"))


def set_display_name(user, name: str) -> None:
    """`set_display_name` sets how other people see *user*; an empty *name*
    shows their email address instead. It raises `.ServiceError` for a name
    under `DISPLAY_NAME_MIN` characters."""
    name = " ".join((name or "").split())
    if name and len(name) < DISPLAY_NAME_MIN:
        raise ServiceError(_("A display name needs at least %(n)s characters.")
                           % {"n": DISPLAY_NAME_MIN})
    account = account_of(user)
    account.display_name = name[:150]
    account.save(update_fields=["display_name"])


def set_active(user, active: bool) -> None:
    """`set_active` blocks or restores sign-in; *user* keeps their experiments."""
    user.is_active = active
    user.save(update_fields=["is_active"])


def delete_user(user) -> None:
    """`delete_user` raises `.ServiceError` when *user* still owns live
    experiments or leads a group with other people in it. Experiments of
    theirs in the bin lose their owner and, with no bin left holding them, go
    to the site bin (`.orphaned`)."""
    if user.experiments.filter(deleted_at__isnull=True, trashed_at__isnull=True).exists():
        raise ServiceError(_(
            "%(who)s still owns experiments, so this account cannot be deleted. "
            "Deactivate it instead, or hand the experiments to somebody else "
            "first.") % {"who": shown_name(user)})
    _refuse_while_primary(user)
    _forget(user)


def _led_groups(user):
    """`_led_groups` lists the memberships in which *user* is primary lead of a
    group with somebody else in it."""
    from access.models import Membership

    return [m for m in user.memberships.filter(role=Membership.PRIMARY_LEAD)
            .select_related("group")
            if m.group.memberships.exclude(user=user).exists()]


def _refuse_while_primary(user) -> None:
    led = _led_groups(user)
    if led:
        raise ServiceError(_(
            "%(who)s is the primary lead of %(groups)s. Make somebody else primary "
            "lead first.") % {"who": shown_name(user),
                              "groups": ", ".join(m.group.name for m in led)})


def successors(user) -> list:
    """`successors` pairs each group *user* leads primarily, with somebody else
    in it, with the membership that takes the role over when *user* leaves:
    the longest-serving group lead, or without one, the member whose account
    is oldest."""
    from access.models import Membership

    pairs = []
    for membership in _led_groups(user):
        others = membership.group.memberships.exclude(user=user).select_related("user")
        heir = (others.filter(role=Membership.LEAD).order_by("joined_at", "pk").first()
                or others.order_by("user__date_joined", "user__pk").first())
        pairs.append((membership, heir))
    return pairs


def erase_user(user) -> None:
    """`erase_user` deletes *user* at their own request (DSGVO Art. 17). Their
    primary-lead role passes to each group's heir from `successors`, and their
    live experiments pass to their group's primary lead, or to the bin when
    nobody else is in the group."""
    from access.models import Membership
    from access.services import groups as group_service
    from experiments.services import bin as bin_service

    with transaction.atomic():
        for membership, heir in successors(user):
            if heir.role != Membership.LEAD:
                heir.role = Membership.LEAD
                heir.save(update_fields=["role"])
            group_service.transfer_primary(membership, heir)
        for experiment in user.experiments.filter(deleted_at__isnull=True,
                                                  trashed_at__isnull=True):
            heir = None
            if experiment.group_id is not None:
                heir = (experiment.group.memberships.exclude(user=user)
                        .filter(role=Membership.PRIMARY_LEAD).select_related("user").first())
            if heir is None:
                bin_service.delete(experiment, by=user)
                continue
            experiment.shares.filter(user=heir.user).delete()
            experiment.owner = heir.user
            experiment.save(update_fields=["owner"])
        _forget(user)


def _forget(user) -> None:
    """`_forget` deletes *user* with what still points at them: their sessions,
    their assistant actions and their failed sign-ins."""
    from django.contrib.sessions.models import Session

    from assistant.models import AgentAction

    pk = str(user.pk)
    for session in Session.objects.filter(expire_date__gt=timezone.now()):
        if session.get_decoded().get("_auth_user_id") == pk:
            session.delete()
    AgentAction.objects.filter(user=user).delete()
    try:
        from axes.models import AccessAttempt, AccessFailureLog, AccessLog
    except ImportError:
        pass
    else:
        names = {user.get_username(), user.email}
        for model in (AccessAttempt, AccessFailureLog, AccessLog):
            model.objects.filter(username__in=names).delete()
    user.delete()


def set_email(user, email: str) -> None:
    """`set_email` changes the address *user* signs in with, which is also
    their username (DSGVO Art. 16). It raises `.ServiceError` for an invalid
    address or one another account uses."""
    from access.services import accounts

    address = accounts.normalize(email)
    try:
        validate_email(address)
    except ValidationError as error:
        raise ServiceError(_("%(address)s is not an email address.") % {"address": email}) \
            from error
    User = get_user_model()
    taken = User.objects.exclude(pk=user.pk).filter(email__iexact=address).exists() or \
        User.objects.exclude(pk=user.pk).filter(username__iexact=address).exists()
    if taken:
        raise ServiceError(_("Another account already uses %(address)s.") % {"address": address})
    user.email = address
    if len(address) <= accounts.USERNAME_MAX:
        user.username = address
    user.save(update_fields=["email", "username"])


def _custom_models_permission():
    return Permission.objects.get(content_type__app_label="access",
                                  codename="use_custom_models")


def set_custom_models(user, allowed: bool) -> None:
    """`set_custom_models` toggles *user*'s permission to upload and run custom
    model code."""
    permission = _custom_models_permission()
    if allowed:
        user.user_permissions.add(permission)
    else:
        user.user_permissions.remove(permission)
