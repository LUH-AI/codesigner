from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core import io

from ui.services import snapshot as adapter


class Command(BaseCommand):
    help = "Import an .ihpo file into the database as an Experiment."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str, help="Path to the .ihpo file")

    def handle(self, *args, **options):
        path = Path(options["path"])
        try:
            snapshot = io.parse(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc))

        exp = adapter.experiment_from_snapshot(snapshot)
        self.stdout.write(self.style.SUCCESS(f"Imported experiment {exp.name!r} (id {exp.pk})"))
