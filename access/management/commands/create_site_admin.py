"""Create somebody who manages groups, and give them one of their own.

A site admin holds `manage_site`: they create groups, set how many people each
may hold, see what each is using, and stop a job that is going wrong. What they
are shown is decided by their membership like anybody else's — managing groups
and being in one are separate facts — so by default they get a group of one with
themselves as its primary lead, and see that group and no other.

    manage.py create_site_admin ana@lab.de            # + a group of their own
    manage.py create_site_admin ana@lab.de --users 0  # no group at all
    manage.py create_site_admin ana@lab.de --users 3  # a group they may fill

The email address is the account's key, as everywhere else — see
`access/services/accounts.py`. The username is derived from it.

`--users` is the new group's seat limit, and defaults to **1**: a site admin
with somewhere to put an experiment of their own, and no room to grow a group
underneath themselves without deciding to. **0 means no group is created**,
which is the cleaner reading of the role and the right answer when they will
never run anything here.

Not an override of `createsuperuser`. `django.contrib.auth` comes before
`access` in INSTALLED_APPS and `get_commands()` resolves so the earlier app
wins, so overriding it would mean reordering INSTALLED_APPS — a strange thing to
do to one list for the sake of one command. Stock `createsuperuser` still works
and makes a superuser with no group, which is a legitimate state.

**Superuser is deliberately not implied.** A Django superuser can read every
experiment through the admin, so a site admin who should genuinely not see
colleagues' work must not be one. `--superuser` is there for the operator who
wants both and knows what they are asking for.
"""

import getpass

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from access.models import Group, Membership


class Command(BaseCommand):
    help = "Create a site admin, with a group of their own to work in."

    def add_arguments(self, parser):
        parser.add_argument("email")
        parser.add_argument(
            "--users", type=int, default=1, metavar="N",
            help="seats in the group created for them; 0 creates no group "
                 "(default: 1)")
        parser.add_argument(
            "--superuser", action="store_true",
            help="also make them a Django superuser. Not the default: a "
                 "superuser can read every experiment through /admin/, which "
                 "is what this role is meant not to do.")
        parser.add_argument(
            "--password", default=None,
            help="for scripts and seeding. Left out, it is prompted for.")

    @transaction.atomic
    def handle(self, *args, **options):
        from django.core.exceptions import ValidationError

        from access.services import accounts

        address = accounts.normalize(options["email"])
        seats = options["users"]

        if seats < 0:
            raise CommandError("--users cannot be negative")
        # Refused rather than promoted. Granting `manage_site` to an account
        # that already exists is a real decision about a real person, and a
        # command called `create_` should not make it as a side effect of a
        # typo'd address that happened to match somebody.
        if accounts.find(address) is not None:
            raise CommandError(f"there is already an account for {address!r}")

        password = options["password"] or getpass.getpass("Password: ")
        if not password:
            raise CommandError("a password is required")

        try:
            user, _, _ = accounts.find_or_create_by_email(address, password=password)
        except ValidationError:
            raise CommandError(f"{address!r} is not an email address")
        # `is_staff` is Django's flag for reaching /admin/, and reaching it is
        # reading every experiment. Off unless they asked to be a superuser,
        # for whom the point is moot.
        user.is_superuser = user.is_staff = options["superuser"]
        user.save(update_fields=["is_superuser", "is_staff"])
        user.user_permissions.add(_manage_site())
        username = user.get_username()

        if seats:
            group = Group.objects.create(name=_free_name(address.split("@", 1)[0]),
                                         user_limit=seats)
            # Primary, not merely lead: they are the only person in it, so
            # there is nobody else who could appoint one, and a group whose
            # leads cannot be appointed is a group that cannot grow.
            Membership.objects.create(user=user, group=group,
                                      role=Membership.PRIMARY_LEAD)
            self.stdout.write(self.style.SUCCESS(
                f"Created site admin {username!r} and group {group.name!r} "
                f"({seats} seat{'s' if seats != 1 else ''})."))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Created site admin {username!r}, in no group."))

        if not options["superuser"]:
            self.stdout.write(
                "They cannot open /admin/, which is deliberate: it would show "
                "them every experiment on the instance.")


def _manage_site() -> Permission:
    return Permission.objects.get(content_type__app_label="access",
                                  codename="manage_site")


def _free_name(base: str) -> str:
    name, n = base, 2
    while Group.objects.filter(name=name).exists():
        name, n = f"{base}-{n}", n + 1
    return name
