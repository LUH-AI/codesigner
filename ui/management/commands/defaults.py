"""`manage.py defaults` shows and changes the settings every experiment
inherits.

    manage.py defaults show
    manage.py defaults set KEY=VALUE [KEY=VALUE ...]
    manage.py defaults reset
"""

from django.core.management.base import CommandError

from experiments.management.jsoncommand import JsonCommand
from experiments.services.settings import coerce_settings, set_global_defaults
from ui.experiment_settings import SETTING_DEFAULTS, global_defaults


class Command(JsonCommand):
    help = "Show and change the instance's default experiment settings."

    def add_subcommands(self, sub):
        sub.add_parser("show")
        change = sub.add_parser("set")
        change.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
        sub.add_parser("reset")

    def do_show(self, options):
        return global_defaults()

    def do_set(self, options):
        values = dict(global_defaults())
        for pair in options["pairs"]:
            key, sep, value = pair.partition("=")
            if not sep or key not in SETTING_DEFAULTS:
                raise CommandError(f"{pair!r} is not KEY=VALUE for a known setting; "
                                   "`defaults show` lists them.")
            if isinstance(SETTING_DEFAULTS[key], bool):
                values[key] = value.strip().lower() in ("1", "true", "on", "yes")
            else:
                values[key] = value
        set_global_defaults(coerce_settings(values, SETTING_DEFAULTS))
        return global_defaults()

    def do_reset(self, options):
        set_global_defaults({})
        return global_defaults()
