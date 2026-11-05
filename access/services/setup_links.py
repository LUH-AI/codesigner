"""A deployment's onboarding decides when to issue a `.SetupLink` and how to
deliver the link.
"""

import hashlib
import secrets
from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from access.models import Account, SetupLink, account_of


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue(user, hours: float) -> str:
    """`issue` returns the new link's path on the site (`SITE_URLCONF`),
    whichever project calls it; earlier unused links of *user* expire at
    once."""
    SetupLink.objects.filter(user=user, used_at__isnull=True).update(
        expires_at=timezone.now())
    token = secrets.token_urlsafe(32)
    SetupLink.objects.create(user=user, token_hash=_hash(token),
                             expires_at=timezone.now() + timedelta(hours=hours))
    account = account_of(user)
    if not user.has_usable_password():
        account.password_pending = Account.SETUP_LINK
        account.save(update_fields=["password_pending"])
    return reverse("ui:account_setup", args=[token], urlconf=settings.SITE_URLCONF)


def live(token: str):
    return (SetupLink.objects.select_related("user")
            .filter(token_hash=_hash(token), used_at__isnull=True,
                    expires_at__gt=timezone.now()).first())


def use(link) -> bool:
    """`use` only returns True for the one request that wins a race for *link*."""
    return bool(SetupLink.objects
                .filter(pk=link.pk, used_at__isnull=True, expires_at__gt=timezone.now())
                .update(used_at=timezone.now()))


def sweep() -> list[str]:
    """`sweep` deletes each waiting account that never signed in and owns no
    experiments or live link, so a lead can add the person again."""
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
        removed.append(user.email or user.get_username())
        user.delete()
    SetupLink.objects.filter(Q(used_at__isnull=False) | Q(expires_at__lte=now)).delete()
    return removed
