"""`manage.py groups` lists, creates, edits and deletes groups.
"""

from django.utils.translation import gettext as _

from access.models import Group
from access.services import groups as group_service
from access.services.errors import ServiceError
from experiments.management.jsoncommand import JsonCommand


def group_named(name_or_pk: str):
    """`group_named` takes a group name or pk."""
    group = Group.objects.filter(name=name_or_pk).first()
    if group is None and str(name_or_pk).isdigit():
        group = Group.objects.filter(pk=int(name_or_pk)).first()
    if group is None:
        raise ServiceError(_("There is no group called “%(name)s”.") % {"name": name_or_pk})
    return group


def group_row(group) -> dict:
    from experiments.services.storage import usage

    storage = usage(group)
    return {"id": group.pk, "name": group.name, "seats": group.user_limit,
            "members": group.memberships.count(), "active": group.is_active,
            "storage_used_bytes": storage["total"], "storage_limit_bytes": storage["limit"]}


class Command(JsonCommand):
    help = "List, create, edit and delete groups."

    def add_subcommands(self, sub):
        sub.add_parser("list")
        create = sub.add_parser("create")
        create.add_argument("name")
        create.add_argument("--seats", type=int, required=True)
        create.add_argument("--lead", required=True,
                            help="The primary lead's email address.")
        update = sub.add_parser("update")
        update.add_argument("group")
        update.add_argument("--name")
        update.add_argument("--seats", type=int)
        update.add_argument("--storage-gb", type=float)
        state = update.add_mutually_exclusive_group()
        state.add_argument("--active", dest="active", action="store_true", default=None)
        state.add_argument("--inactive", dest="active", action="store_false")
        delete = sub.add_parser("delete")
        delete.add_argument("group")

    def do_list(self, options):
        return [group_row(g) for g in Group.objects.order_by("name")]

    def do_create(self, options):
        group, lead, message = group_service.create_group(
            options["name"], options["seats"], options["lead"])
        return {**group_row(group), "lead": lead.email, "message": message}

    def do_update(self, options):
        group = group_named(options["group"])
        group_service.update_group(
            group, name=options["name"] or group.name,
            seats=group.user_limit if options["seats"] is None else options["seats"],
            active=group.is_active if options["active"] is None else options["active"],
            storage_gb=options["storage_gb"])
        return group_row(group)

    def do_delete(self, options):
        group = group_named(options["group"])
        group_service.delete_group(group)
        return {"deleted": group.name}
