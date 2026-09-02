import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core import io, provenance, smac_import

from ui.services import snapshot as adapter


class Command(BaseCommand):
    help = "Import an .ihpo file, or a SMAC output directory, as an Experiment."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str,
                            help="Path to the .ihpo file, or to a SMAC run "
                                 "directory when --smac is given")
        parser.add_argument(
            "--smac", action="store_true",
            help="Read PATH as a SMAC output directory (the folder holding its "
                 "runhistory.json) rather than as an .ihpo file. Imported "
                 "read-only: SMAC records nothing about what was optimized.")

    def handle(self, *args, **options):
        path = Path(options["path"])
        if options["smac"]:
            return self._import_smac(path)
        try:
            snapshot = io.parse(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc))

        # The paths inside the file are adopted, so the dataset they name has to
        # be the one the trials were measured on — otherwise the experiment
        # resumes against different data and every later trial is compared with
        # a history it does not belong to.
        stored = (snapshot.get("dataset") or {}).get("path", "")
        if stored and Path(stored).is_file():
            mismatch = provenance.dataset_mismatch(
                snapshot.get("dataset"), provenance.sha256(stored))
            if mismatch:
                raise CommandError(mismatch)

        # A command run by whoever operates the server, naming a file they chose,
        # so the dataset and model paths inside it are theirs to adopt. The web
        # importer deliberately does not do this — see experiment_from_snapshot.
        exp = adapter.experiment_from_snapshot(snapshot, adopt_paths=True)
        self._report(exp)

    def _import_smac(self, directory: Path) -> None:
        """Read a SMAC run directory. No paths to adopt — there are none in it.

        Recursive, so either the run directory or the `smac3_output` above it
        works, which is the same latitude the web importer allows. The importer
        matches on basenames, so two runs at once cannot be told apart — refused
        here for the same reason it is refused there.
        """
        if not directory.is_dir():
            raise CommandError(f"not a directory: {directory}")

        files = {}
        for found in sorted(directory.rglob("*.json")):
            if found.name in files:
                raise CommandError(
                    f"two files here are called {found.name} — this looks like "
                    "more than one run. Name a single run directory.")
            try:
                files[found.name] = json.loads(found.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CommandError(f"could not read {found}: {exc}")

        try:
            snapshot = io.parse(io.to_bytes(smac_import.snapshot_from_smac(files)))
        except ValueError as exc:
            raise CommandError(str(exc))

        self._report(adapter.experiment_from_snapshot(snapshot))

    def _report(self, exp) -> None:
        self.stdout.write(self.style.SUCCESS(f"Imported experiment {exp.name!r} (id {exp.pk})"))
