"""`JsonCommand` is the base of the site-management commands, and its `--json`
switch prints output for scripts.
"""

import argparse
import json

from django.core.management.base import BaseCommand, CommandError


class JsonCommand(BaseCommand):
    """`JsonCommand` runs `do_<subcommand>` and hands its result to `emit`."""

    #: Switches also valid after the subcommand.
    switches = (("--json", "Print JSON."),)

    def add_arguments(self, parser):
        for flag, text in self.switches:
            parser.add_argument(flag, action="store_true", help=text)
        sub = parser.add_subparsers(dest="subcommand", required=True)
        self.add_subcommands(sub)
        # `SUPPRESS` keeps a switch set before the subcommand name.
        for subparser in sub.choices.values():
            for flag, text in self.switches:
                subparser.add_argument(flag, action="store_true", help=text,
                                       default=argparse.SUPPRESS)

    def add_subcommands(self, sub):
        raise NotImplementedError

    def handle(self, *args, **options):
        from access.services.errors import ServiceError

        handler = getattr(self, "do_" + options["subcommand"].replace("-", "_"))
        try:
            result = handler(options)
        except ServiceError as error:
            raise CommandError(str(error))
        self.emit(result, options["json"])

    def emit(self, result, as_json: bool) -> None:
        if result is None:
            return
        if as_json:
            self.stdout.write(json.dumps(result, default=str, indent=2))
            return
        if isinstance(result, str):
            self.stdout.write(result)
        elif isinstance(result, dict):
            for key, value in result.items():
                self.stdout.write(f"{key}: {value}")
        elif result:
            columns = list(result[0])
            widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in result))
                      for c in columns}
            self.stdout.write("  ".join(c.ljust(widths[c]) for c in columns))
            for row in result:
                self.stdout.write("  ".join(str(row.get(c, "")).ljust(widths[c])
                                            for c in columns))
