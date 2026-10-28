"""`create_site_admin` makes an account holding `manage_site` and, by default, a
one-seat group with the new admin as primary lead. Membership decides what a
site admin sees, as for anybody else, and `--users 0` creates no group. The
command leaves `createsuperuser` alone, since overriding it would mean
reordering INSTALLED_APPS. `--superuser` stays opt-in because a superuser can
read every experiment through the admin.
"""

import getpass

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
        # An existing account is refused, so a typo'd address never grants
        # `manage_site` to somebody else.
        if accounts.find(address) is not None:
            raise CommandError(f"there is already an account for {address!r}")

        password = options["password"] or getpass.getpass("Password: ")
        if not password:
            raise CommandError("a password is required")

        try:
            user, _, _ = accounts.find_or_create_by_email(address, password=password)
        except ValidationError:
            raise CommandError(f"{address!r} is not an email address")
        # `is_staff` opens /admin/ and every experiment there, so only
        # `--superuser` sets it.
        user.is_superuser = user.is_staff = options["superuser"]
        user.save(update_fields=["is_superuser", "is_staff"])
        user.user_permissions.add(_manage_site())
        username = user.get_username()

        if seats:
            group = Group.objects.create(name=_free_name(address.split("@", 1)[0]),
                                         user_limit=seats)
            # The sole member is primary lead, since nobody else could appoint
            # leads.
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
