"""`access.services.accounts` finds and creates accounts by email address, for
an onboarding that identifies people by address (`access.onboarding`). A
case-insensitive unique index (`access/migrations/0009`) stops two racing
requests from creating two accounts for one address.
"""

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


def find_or_create_by_email(email: str):
    """`find_or_create_by_email` returns `(user, created)`. A new account has
    no usable password until its person sets one. Its username, which nobody
    sees or types, is a copy of the address, or a free name from the part
    before the `@` in the rare case another account holds the address as its
    username. An invalid address raises `ValidationError` before anything is
    written.
    """
    address = normalize(email)
    validate_email(address)

    existing = find(address)
    if existing is not None:
        return existing, False

    User = get_user_model()
    if len(address) <= USERNAME_MAX and not User.objects.filter(
            username__iexact=address).exists():
        username = address
    else:
        username = free_username(address.split("@", 1)[0])

    with transaction.atomic():
        user = User.objects.create_user(username=username, email=address)
        user.set_unusable_password()
        user.save(update_fields=["password"])
    return user, True
