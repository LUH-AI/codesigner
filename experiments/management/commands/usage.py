"""`manage.py usage` reports trials, runs and trial seconds per group, per
person or per model.

    manage.py usage groups|people|models [--csv]
"""

import csv

from experiments.management.jsoncommand import JsonCommand
from experiments.services import site as site_service


def group_rows():
    return [{"group": g.name, "members": g.members, "trials": g.trials or 0,
             "runs": g.runs, "trial_seconds": round(g.seconds or 0, 1)}
            for g in site_service.usage_by_group().order_by("name")]


def person_rows():
    return [{"group": m.group.name, "email": m.user.email, "role": m.role,
             "trials": m.trials or 0, "runs": m.runs,
             "trial_seconds": round(m.seconds or 0, 1)}
            for m in site_service.usage_by_person()]


class Command(JsonCommand):
    help = "Report usage per group, person or model."

    switches = (*JsonCommand.switches, ("--csv", "Print CSV."))

    def add_subcommands(self, sub):
        sub.add_parser("groups")
        sub.add_parser("people")
        sub.add_parser("models")

    def do_groups(self, options):
        return self._maybe_csv(group_rows(), options)

    def do_people(self, options):
        return self._maybe_csv(person_rows(), options)

    def do_models(self, options):
        rows = [{**row, "seconds": round(row["seconds"], 1)}
                for row in site_service.usage_by_model()]
        return self._maybe_csv(rows, options)
    def _maybe_csv(self, rows, options):
        if not options["csv"]:
            return rows
        if rows:
            writer = csv.DictWriter(self.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return None
