"""Codesigner's views and commands create every `User` through
`find_or_create_by_email`, which keeps the email address a reliable key. A
case-insensitive unique index (`access/migrations/0009`) backs that key up
against races, and a duplicate typed into the Django admin fails with an
`IntegrityError`.
"""

import secrets

from django.contrib.auth import get_user_model
from django.core.validators import validate_email
from django.db import transaction

#: Limit from `AbstractUser.username`.
USERNAME_MAX = 150


def normalize(email: str) -> str:
    """`normalize` folds the local part to lower case despite RFC 5321, because
    `Ana@lab.de` and `ana@lab.de` must be one account.
    """
    return (email or "").strip().lower()


def find(email: str):
    """`find` returns *email*'s account or None."""
    address = normalize(email)
    if not address:
        return None
    return get_user_model().objects.filter(email__iexact=address).first()


def free_username(base: str) -> str:
    """*base* as a free username, suffixed `-2`, `-3` while taken."""
    User = get_user_model()
    base = base[:USERNAME_MAX] or "user"
    name, n = base, 2
    while User.objects.filter(username__iexact=name).exists():
        suffix = f"-{n}"
        name, n = base[:USERNAME_MAX - len(suffix)] + suffix, n + 1
    return name


def generated_password() -> str:
    """`generated_password` makes a short first password to hand over in person,
    since no mail server exists.
    """
    return secrets.token_urlsafe(9)


def find_or_create_by_email(email: str, *, password: str = ""):
    """`find_or_create_by_email` returns `(user, created, password)`, where
    *password* is the one given or generated for a new account and empty for an
    existing one. The username is the address itself if it fits and is free,
    else a free name from the part before the `@`. An invalid address raises
    `ValidationError` before anything is written.
    """
    address = normalize(email)
    validate_email(address)

    existing = find(address)
    if existing is not None:
        return existing, False, ""

    User = get_user_model()
    if len(address) <= USERNAME_MAX and not User.objects.filter(
            username__iexact=address).exists():
        username = address
    else:
        username = free_username(address.split("@", 1)[0])

    given = password or generated_password()
    with transaction.atomic():
        user = User.objects.create_user(username=username, email=address,
                                        password=given)
    return user, True, given
