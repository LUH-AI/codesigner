from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core import io, provenance

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

        # The paths inside the file are adopted, so the dataset they name has to
        # be the one the trials were measured on — otherwise the experiment
        # resumes against different data and every later trial is compared with
        # a history it does not belong to.
        stored = snapshot.get("dataset_path", "")
        if stored and Path(stored).is_file():
            mismatch = provenance.dataset_mismatch(
                snapshot.get("data"), provenance.sha256(stored))
            if mismatch:
                raise CommandError(mismatch)

        # A command run by whoever operates the server, naming a file they chose,
        # so the dataset and model paths inside it are theirs to adopt. The web
        # importer deliberately does not do this — see experiment_from_snapshot.
        exp = adapter.experiment_from_snapshot(snapshot, adopt_paths=True)
        self.stdout.write(self.style.SUCCESS(f"Imported experiment {exp.name!r} (id {exp.pk})"))
