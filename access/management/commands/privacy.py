"""`manage.py privacy sweep` deletes personal data past its retention
(`access.services.retention`). A timer runs it every hour.

    manage.py privacy sweep [--dry-run]
"""

from access.services import retention
from experiments.management.jsoncommand import JsonCommand


class Command(JsonCommand):
    help = "Delete personal data past its retention."

    def add_subcommands(self, sub):
        sweep = sub.add_parser("sweep")
        sweep.add_argument("--dry-run", action="store_true",
                           help="Count what is due without deleting it.")

    def do_sweep(self, options):
        return retention.sweep(dry_run=options["dry_run"])
