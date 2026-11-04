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
from django.contrib.auth import get_user_model
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError
from django.utils.module_loading import import_string
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from access.models import Account, account_of
from access.services.errors import ServiceError


class LocalOnboarding:
    """`LocalOnboarding` lets the first sign-in set the password."""

    label = gettext_lazy("Username")
    input_type = "text"
    offers_reset = False

    def normalize(self, identifier: str) -> str:
        return (identifier or "").strip()

    def find(self, identifier: str):
        """`find` ignores usernames that break `UnicodeUsernameValidator`, which
        leaves system accounts such as codesigner-manage's out of reach."""
        identifier = self.normalize(identifier)
        User = get_user_model()
        try:
            UnicodeUsernameValidator()(identifier)
        except ValidationError:
            return None
        return (User.objects.filter(username__iexact=identifier).first()
                or User.objects.filter(email__iexact=identifier).first())

    def create(self, identifier: str):
        username = self.normalize(identifier)
        if not username or len(username) > 150:
            raise ServiceError(_("A username of 1 to 150 characters is needed."))
        try:
            UnicodeUsernameValidator()(username)
        except ValidationError:
            raise ServiceError(_("A username can only contain letters, digits and @ . + - _."))
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
    path = getattr(settings, "ACCOUNT_ONBOARDING", "") or "access.onboarding.LocalOnboarding"
    return import_string(path)()


def agreement():
    """`agreement` returns `(version, html)` from `ACCOUNT_AGREEMENT`, or None."""
    path = getattr(settings, "ACCOUNT_AGREEMENT", "")
    return import_string(path)() if path else None
