"""Passwords and the agreement: what every way of setting a password records,
and what `.AccountGateMiddleware` asks of a signed-in person.
"""

from django.utils import timezone

from access.models import Account, AgreementAcceptance, account_of
from access.onboarding import agreement


def password_set(user) -> None:
    """`password_set` records that *user* chose a password: nothing is pending
    and no change is required. The caller saves the password itself."""
    account = account_of(user)
    account.password_pending = ""
    account.must_change_password = False
    account.password_changed_at = timezone.now()
    account.save(update_fields=["password_pending", "must_change_password",
                                "password_changed_at"])


def require_new_password(users) -> int:
    """`require_new_password` makes each of *users* set a new password at their
    next request, and returns how many."""
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
    """`agreement_due` returns `(version, html)` when *user* has not accepted
    the current agreement, else None."""
    current = agreement()
    if current is None:
        return None
    if AgreementAcceptance.objects.filter(user=user, version=current[0]).exists():
        return None
    return current


def accept_agreement(user, version: str) -> None:
    AgreementAcceptance.objects.get_or_create(user=user, version=version)
