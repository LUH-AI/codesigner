"""Create somebody who manages groups rather than experiments.

A site admin holds `manage_site`: they create groups, set how many people each
may hold, see what each is using, and stop a job that is going wrong. They are
shown no experiments — that is the separation the role exists for.

    manage.py create_site_admin alice            # + a group of their own
    manage.py create_site_admin alice --users 0  # no group at all
    manage.py create_site_admin alice --users 3  # a group they may fill

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
    help = "Create a site admin, who manages groups and sees no experiments."

    def add_arguments(self, parser):
        parser.add_argument("username")
        parser.add_argument("--email", default="")
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
        User = get_user_model()
        username = options["username"]
        seats = options["users"]

        if seats < 0:
            raise CommandError("--users cannot be negative")
        if User.objects.filter(username=username).exists():
            raise CommandError(f"there is already a user called {username!r}")

        password = options["password"] or getpass.getpass("Password: ")
        if not password:
            raise CommandError("a password is required")

        user = User.objects.create_user(
            username=username, email=options["email"], password=password,
            is_superuser=options["superuser"],
            # `is_staff` is Django's flag for reaching /admin/, and reaching it
            # is reading every experiment. Off unless they asked to be a
            # superuser, for whom the point is moot.
            is_staff=options["superuser"])
        user.user_permissions.add(_manage_site())

        if seats:
            group = Group.objects.create(name=_free_name(username),
                                         user_limit=seats)
            Membership.objects.create(user=user, group=group,
                                      role=Membership.LEAD)
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
