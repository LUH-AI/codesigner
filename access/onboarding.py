"""`ACCOUNT_ONBOARDING` names the class that turns a lead's "add somebody" into
an account: `LocalOnboarding` by default, or a class that mails a `SetupLink`.

`label` and `input_type` describe the identifier field. `find(identifier)`
returns the account or None, `create(identifier)` makes one without a usable
password, and `welcome(user, created, group)` runs once the membership exists
and returns the message for whoever added *user*. `reset(user)` makes the
password settable again and returns the operator's message; `offers_reset`
decides whether the sign-in page links to `ui:password_forgot`, which calls
`reset`.
"""

from django.conf import settings
from django.utils.module_loading import import_string
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from access.models import Account, account_of, shown_name
from access.services import accounts


class LocalOnboarding:
    """`LocalOnboarding` adds people by email address, sends nothing, and lets
    the first sign-in with that address set the password."""

    label = gettext_lazy("Email address")
    input_type = "email"
    offers_reset = False

    def normalize(self, identifier: str) -> str:
        return accounts.normalize(identifier)

    def find(self, identifier: str):
        return accounts.find(identifier)

    def create(self, identifier: str):
        """`create` raises `ValidationError` for something other than an email
        address."""
        user, _created = accounts.find_or_create_by_email(identifier)
        Account.objects.update_or_create(
            user=user, defaults={"password_pending": Account.FIRST_SIGN_IN})
        return user

    def welcome(self, user, created: bool, group) -> str:
        if created:
            return _("Added %(name)s to %(group)s. They choose their password the first "
                     "time they sign in with this email address.") % {
                         "name": user.email, "group": group.name}
        return _("Added %(name)s to %(group)s.") % {"name": shown_name(user),
                                                   "group": group.name}

    def reset(self, user) -> str:
        user.set_unusable_password()
        user.save(update_fields=["password"])
        account = account_of(user)
        account.password_pending = Account.FIRST_SIGN_IN
        account.save(update_fields=["password_pending"])
        return _("%(name)s chooses a new password the next time they sign in.") % {
            "name": shown_name(user)}


def onboarding():
    path = getattr(settings, "ACCOUNT_ONBOARDING", "") or "access.onboarding.LocalOnboarding"
    return import_string(path)()


def agreement():
    """`agreement` returns `(version, html)` from `ACCOUNT_AGREEMENT`, or None."""
    path = getattr(settings, "ACCOUNT_AGREEMENT", "")
    return import_string(path)() if path else None
