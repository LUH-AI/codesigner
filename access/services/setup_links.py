"""One-time links to set a password (`.SetupLink`). Codesigner issues them and
serves the page they open (`ui:account_setup`); a deployment's onboarding
decides when to issue one, for how long, and how it reaches the person.
"""

import hashlib
import secrets
from datetime import timedelta

from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from access.models import Account, SetupLink, account_of


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue(user, hours: float) -> str:
    """`issue` creates a link for *user* valid for *hours* and returns its
    path. Earlier unused links of *user* stop working, so only the newest
    counts."""
    SetupLink.objects.filter(user=user, used_at__isnull=True).update(
        expires_at=timezone.now())
    token = secrets.token_urlsafe(32)
    SetupLink.objects.create(user=user, token_hash=_hash(token),
                             expires_at=timezone.now() + timedelta(hours=hours))
    account = account_of(user)
    if not user.has_usable_password():
        account.password_pending = Account.SETUP_LINK
        account.save(update_fields=["password_pending"])
    return reverse("ui:account_setup", args=[token])


def live(token: str):
    """`live` returns the unused, unexpired link for *token*, or None."""
    return (SetupLink.objects.select_related("user")
            .filter(token_hash=_hash(token), used_at__isnull=True,
                    expires_at__gt=timezone.now()).first())


def use(link) -> bool:
    """`use` marks *link* used, and is True only for the one request that
    does: two requests racing for one link cannot both succeed."""
    return bool(SetupLink.objects
                .filter(pk=link.pk, used_at__isnull=True, expires_at__gt=timezone.now())
                .update(used_at=timezone.now()))


def sweep() -> list[str]:
    """`sweep` deletes accounts that were waiting for a setup link whose links
    all expired unused, with their memberships, so a lead can add them again;
    it returns their usernames. Only an account that never signed in and owns
    nothing goes. Used and expired links are deleted too."""
    now = timezone.now()
    waiting = (Account.objects.filter(password_pending=Account.SETUP_LINK,
                                      user__last_login__isnull=True)
               .select_related("user"))
    removed = []
    for account in waiting:
        user = account.user
        if (user.has_usable_password() or user.experiments.exists()
                or user.setup_links.filter(expires_at__gt=now).exists()):
            continue
        removed.append(user.get_username())
        user.delete()
    SetupLink.objects.filter(Q(used_at__isnull=False) | Q(expires_at__lte=now)).delete()
    return removed
