"""`manage.py usage` reports runs, trials and trial seconds per group or per
person within each group.

    manage.py usage groups [--json | --csv]
    manage.py usage people [--json | --csv]
"""

import csv

from experiments.management.jsoncommand import JsonCommand
from experiments.services import site as site_service


def group_rows():
    return [{"group": g.name, "members": g.members, "runs": g.runs,
             "trials": g.trials or 0, "trial_seconds": round(g.seconds or 0, 1)}
            for g in site_service.usage_by_group().order_by("name")]


def person_rows():
    return [{"group": m.group.name, "email": m.user.email, "role": m.role,
             "runs": m.runs, "trials": m.trials or 0,
             "trial_seconds": round(m.seconds or 0, 1)}
            for m in site_service.usage_by_person()]


class Command(JsonCommand):
    help = "Report usage per group or per person."

    switches = (*JsonCommand.switches, ("--csv", "Print CSV."))

    def add_subcommands(self, sub):
        sub.add_parser("groups")
        sub.add_parser("people")

    def do_groups(self, options):
        return self._maybe_csv(group_rows(), options)

    def do_people(self, options):
        return self._maybe_csv(person_rows(), options)

    def _maybe_csv(self, rows, options):
        if not options["csv"]:
            return rows
        if rows:
            writer = csv.DictWriter(self.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return None
