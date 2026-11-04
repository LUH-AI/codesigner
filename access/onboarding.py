"""How a lead's "add somebody" becomes an account, and how a forgotten password
is set again. `ACCOUNT_ONBOARDING` names the class; `LocalOnboarding` is the
default for an install without email: a lead adds a username, and the person
chooses a password the first time they sign in with it.

A deployment replaces the class to change the identifier and what follows
adding somebody, for example requiring an email address and mailing a
`SetupLink` (`access.services.setup_links`). Its methods:

- `label` and `input_type` describe the identifier field on the group page;
- `find(identifier)` returns the existing account, or None;
- `create(identifier)` makes the account, without a usable password;
- `welcome(user, created, group)` runs after the membership exists and
  returns the message for the person who added them;
- `reset(user)` makes the account's password settable again and returns a
  message for the operator; `offers_reset` says whether the sign-in page links
  to `ui:password_forgot`, whose page calls `reset` for the identifier typed.
"""

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.module_loading import import_string
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from access.models import Account, account_of
from access.services.errors import ServiceError


class LocalOnboarding:
    """`LocalOnboarding` adds people by username; their first sign-in sets the
    password, which suits a trusted network without a mail server."""

    label = gettext_lazy("Username")
    input_type = "text"
    offers_reset = False

    def normalize(self, identifier: str) -> str:
        return (identifier or "").strip()

    def find(self, identifier: str):
        identifier = self.normalize(identifier)
        User = get_user_model()
        return (User.objects.filter(username__iexact=identifier).first()
                or User.objects.filter(email__iexact=identifier).first())

    def create(self, identifier: str):
        username = self.normalize(identifier)
        if not username or len(username) > 150:
            raise ServiceError(_("A username of 1 to 150 characters is needed."))
        user = get_user_model().objects.create_user(username=username)
        user.set_unusable_password()
        user.save(update_fields=["password"])
        Account.objects.update_or_create(
            user=user, defaults={"password_pending": Account.FIRST_SIGN_IN})
        return user

    def welcome(self, user, created: bool, group) -> str:
        if created:
            return _("Added %(name)s to %(group)s. They choose their password the first "
                     "time they sign in with this username.") % {
                         "name": user.get_username(), "group": group.name}
        return _("Added %(name)s to %(group)s.") % {"name": user.get_username(),
                                                   "group": group.name}

    def reset(self, user) -> str:
        user.set_unusable_password()
        user.save(update_fields=["password"])
        account = account_of(user)
        account.password_pending = Account.FIRST_SIGN_IN
        account.save(update_fields=["password_pending"])
        return _("%(name)s chooses a new password the next time they sign in.") % {
            "name": user.get_username()}


def onboarding():
    """`onboarding` returns an instance of the class `ACCOUNT_ONBOARDING`
    names."""
    path = getattr(settings, "ACCOUNT_ONBOARDING", "") or "access.onboarding.LocalOnboarding"
    return import_string(path)()


def agreement():
    """`agreement` returns `(version, html)` from the callable
    `ACCOUNT_AGREEMENT` names, or None when there is no agreement to accept."""
    path = getattr(settings, "ACCOUNT_AGREEMENT", "")
    return import_string(path)() if path else None
