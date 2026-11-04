"""`manage.py members` manages the people of one group.
"""

from django.utils.translation import gettext as _

from access.models import Membership
from access.services import groups as group_service
from access.backends import account_named
from access.services.errors import ServiceError
from experiments.management.jsoncommand import JsonCommand

from .groups import group_named


def membership_of(group, who: str):
    """`membership_of` raises `.ServiceError` unless *who* belongs to *group*."""
    user = account_named(who)
    membership = (group.memberships.select_related("user").filter(user=user).first()
                  if user else None)
    if membership is None:
        raise ServiceError(_("%(who)s is not in %(group)s.")
                           % {"who": who, "group": group.name})
    return membership


def member_row(membership) -> dict:
    return {"email": membership.user.email, "username": membership.user.get_username(),
            "role": membership.role, "joined_at": membership.joined_at}


class Command(JsonCommand):
    help = "Manage the people of a group."

    def add_subcommands(self, sub):
        listing = sub.add_parser("list")
        listing.add_argument("group")
        add = sub.add_parser("add")
        add.add_argument("group")
        add.add_argument("who", help="A username, or what the onboarding takes.")
        add.add_argument("--role", default=Membership.MEMBER,
                         choices=[Membership.MEMBER, Membership.LEAD, Membership.PRIMARY_LEAD],
                         help="primary_lead only for a group without one.")
        remove = sub.add_parser("remove")
        remove.add_argument("group")
        remove.add_argument("who")
        role = sub.add_parser("set-role")
        role.add_argument("group")
        role.add_argument("who")
        role.add_argument("role", choices=[Membership.MEMBER, Membership.LEAD])
        transfer = sub.add_parser("transfer")
        transfer.add_argument("group")
        transfer.add_argument("who")

    def do_list(self, options):
        group = group_named(options["group"])
        return [member_row(m) for m in
                group.memberships.select_related("user").order_by("user__username")]

    def do_add(self, options):
        group = group_named(options["group"])
        user, created, message = group_service.add_person(group, options["who"], options["role"])
        return {"who": user.get_username(), "group": group.name, "role": options["role"],
                "created": created, "message": message}

    def do_remove(self, options):
        group = group_named(options["group"])
        group_service.remove_person(membership_of(group, options["who"]))
        return {"removed": options["who"], "group": group.name}

    def do_set_role(self, options):
        group = group_named(options["group"])
        membership = membership_of(group, options["who"])
        group_service.set_role(membership, options["role"])
        return member_row(membership)

    def do_transfer(self, options):
        group = group_named(options["group"])
        current = group.memberships.filter(role=Membership.PRIMARY_LEAD).first()
        target = membership_of(group, options["who"])
        if current is None:
            target.role = Membership.PRIMARY_LEAD
            target.save(update_fields=["role"])
        else:
            group_service.transfer_primary(current, target)
        return member_row(target)
