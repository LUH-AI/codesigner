"""A function in `users` changes nothing when it raises `.ServiceError`, whose
message gives the reason.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Count
from django.utils.translation import gettext as _

from access.backends import account_named
from access.services.errors import ServiceError


def named(who: str):
    """`named` raises `.ServiceError` when no account matches *who*."""
    user = account_named(who)
    if user is None:
        raise ServiceError(_("There is no account for %(who)s.") % {"who": who})
    return user


def list_users():
    return (get_user_model().objects
            .annotate(experiment_count=Count("experiments"))
            .prefetch_related("memberships__group")
            .order_by("username"))


def set_active(user, active: bool) -> None:
    """`set_active` blocks or restores sign-in; *user* keeps their experiments."""
    user.is_active = active
    user.save(update_fields=["is_active"])


def delete_user(user) -> None:
    """`delete_user` raises `.ServiceError` when *user* still owns experiments."""
    if user.experiments.exists():
        raise ServiceError(_(
            "%(who)s still owns experiments, so this account cannot be deleted. "
            "Deactivate it instead, or hand the experiments to somebody else "
            "first.") % {"who": user.get_username()})
    user.delete()


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
