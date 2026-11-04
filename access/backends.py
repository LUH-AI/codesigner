"""`SiteBackend` signs people in to the site by username or email address.
Staff and superuser accounts belong to the operators' tools, so neither signs
in here nor keeps a session.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend


def account_named(identifier: str):
    """`account_named` finds the account whose username, or else whose unique
    email address, is *identifier*, ignoring case."""
    identifier = (identifier or "").strip()
    if not identifier:
        return None
    User = get_user_model()
    user = User.objects.filter(username__iexact=identifier).first()
    if user is None:
        matches = list(User.objects.filter(email__iexact=identifier)[:2])
        user = matches[0] if len(matches) == 1 else None
    return user


def operator(user) -> bool:
    return bool(user.is_staff or user.is_superuser)


class SiteBackend(ModelBackend):
    """`SiteBackend` is `ModelBackend` with `account_named` lookup, and with
    staff and superuser accounts excluded."""

    def authenticate(self, request, username=None, password=None, **kwargs):
        user = account_named(username)
        if user is None or password is None:
            # The hash still runs, so timing does not reveal unknown names.
            get_user_model()().set_password(password or "")
            return None
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None

    def user_can_authenticate(self, user):
        return super().user_can_authenticate(user) and not operator(user)
