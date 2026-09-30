"""How a person is found, and how an account comes to exist.

There is still no registration surface and no mail server — see `access/urls.py`
— so an account exists because somebody with a group decided it should. This
module is the only door: every path that creates a `User` comes through
`find_or_create_by_email`, which is what makes an email address a reliable key.

The database backs that up with a case-insensitive unique index
(`access/migrations/0009`), but the index is the backstop for a race, not the
enforcement. `email` belongs to `django.contrib.auth.User`, whose migrations this
project does not own, so no model field here can say "unique" — and a form built
over `User` (the Django admin's user editor, for one) will not pre-validate it. A
duplicate typed there surfaces as an `IntegrityError`, which on an operator-only
surface is an acceptable way to be told.
"""

import secrets

from django.contrib.auth import get_user_model
from django.core.validators import validate_email
from django.db import transaction

#: How long a username may be, per `AbstractUser.username`.
USERNAME_MAX = 150


def normalize(email: str) -> str:
    """The form an address is stored and compared in.

    Lower-cased in full, not only the domain. RFC 5321 leaves the local part to
    the receiving server and permits it to be case-sensitive; in practice no
    provider treats it so, and a system where `Ana@lab.de` and `ana@lab.de` are
    two accounts is one where a lead adds the same colleague twice and neither
    copy can see what the other was given.
    """
    return (email or "").strip().lower()


def find(email: str):
    """The account behind *email*, or None."""
    address = normalize(email)
    if not address:
        return None
    return get_user_model().objects.filter(email__iexact=address).first()


def free_username(base: str) -> str:
    """*base* as a username, with a number appended if it is taken.

    The same shape as the free group name in `create_site_admin`, for the same
    question: what to call a thing whose obvious name is already in use.
    """
    User = get_user_model()
    base = base[:USERNAME_MAX] or "user"
    name, n = base, 2
    while User.objects.filter(username__iexact=name).exists():
        suffix = f"-{n}"
        name, n = base[:USERNAME_MAX - len(suffix)] + suffix, n + 1
    return name


def generated_password() -> str:
    """A first password to hand over in person.

    Readable aloud and short enough to type from a screen, which is how it
    travels — there is no mail server to send it by.
    """
    return secrets.token_urlsafe(9)


def find_or_create_by_email(email: str, *, password: str = ""):
    """The account behind *email*, creating one if there is none.

    Returns `(user, created, password)`. *password* is what the new account was
    given — the one passed in, or a generated one when none was — and is empty
    when nothing was created, because an existing account's password is not
    this code's to know or to change.

    The username is the address itself where it fits and is free. That is the
    least surprising name to sign in with, and it cannot collide with another
    address because addresses are unique. An older account may already hold
    that string as a username, so the fallback is a free name built from the
    part before the `@`.

    Raises `ValidationError` for something that is not an address, before
    anything is written.
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
