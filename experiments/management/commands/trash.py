"""`manage.py trash` rehomes or discards site-bin experiments by identifier.
"""

from django.utils.translation import gettext as _

from access.management.commands.groups import group_named
from access.services.errors import ServiceError
from experiments.management.jsoncommand import JsonCommand
from experiments.services import site as site_service


def binned(identifier: str):
    """`binned` raises `.ServiceError` when absent."""
    found = site_service.trash().filter(identifier=identifier).first()
    if found is None:
        raise ServiceError(_("Experiment %(id)s is not in the site bin.") % {"id": identifier})
    return found


class Command(JsonCommand):
    help = "List, rehome and discard the site bin's experiments."

    def add_subcommands(self, sub):
        sub.add_parser("list")
        rehome = sub.add_parser("rehome")
        rehome.add_argument("identifier")
        rehome.add_argument("group")
        sub.add_parser("discard").add_argument("identifier")

    def do_list(self, options):
        return [{"identifier": e.identifier, "owner": e.owner.email if e.owner else "",
                 "runs": e.run_count, "binned_at": e.deleted_at or e.trashed_at}
                for e in site_service.trash_rows()]

    def do_rehome(self, options):
        experiment = binned(options["identifier"])
        group = group_named(options["group"])
        site_service.rehome(experiment, group)
        return {"rehomed": experiment.identifier, "group": group.name}

    def do_discard(self, options):
        experiment = binned(options["identifier"])
        site_service.discard(experiment)
        return {"discarded": options["identifier"]}
