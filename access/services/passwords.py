"""`passwords` records password changes and supplies the checks of
`.AccountGateMiddleware`.
"""

from django.utils import timezone

from access.models import Account, AgreementAcceptance, account_of
from access.onboarding import agreement


def password_set(user) -> None:
    """`password_set` clears `password_pending` and `must_change_password`; the
    caller saves the password itself."""
    account = account_of(user)
    account.password_pending = ""
    account.must_change_password = False
    account.password_changed_at = timezone.now()
    account.save(update_fields=["password_pending", "must_change_password",
                                "password_changed_at"])


def require_new_password(users) -> int:
    count = 0
    for user in users:
        account = account_of(user)
        account.must_change_password = True
        account.save(update_fields=["must_change_password"])
        count += 1
    return count


def must_change_password(user) -> bool:
    return Account.objects.filter(user=user, must_change_password=True).exists()


def agreement_due(user):
    """`agreement_due` returns the `(version, html)` agreement *user* has yet to
    accept, or None."""
    current = agreement()
    if current is None:
        return None
    if AgreementAcceptance.objects.filter(user=user, version=current[0]).exists():
        return None
    return current


def accept_agreement(user, version: str) -> None:
    AgreementAcceptance.objects.get_or_create(user=user, version=version)
