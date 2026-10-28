import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core import io, provenance, smac_import

from experiments.services import snapshot as adapter


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

        # Adopted paths must name the dataset the trials ran on.
        stored = (snapshot.get("dataset") or {}).get("path", "")
        if stored and Path(stored).is_file():
            mismatch = provenance.dataset_mismatch(
                snapshot.get("dataset"), provenance.sha256(stored))
            if mismatch:
                raise CommandError(mismatch)

        # The operator chose the file, so the command adopts its paths, unlike
        # the web importer.
        exp = adapter.experiment_from_snapshot(snapshot, adopt_paths=True)
        adapter.finish_import(exp, source="ihpo", filename=path.name)
        self._report(exp)

    def _import_smac(self, directory: Path) -> None:
        """`_import_smac` accepts a run directory or its `smac3_output` parent
        and refuses duplicate basenames, the mark of two runs.
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

        exp = adapter.experiment_from_snapshot(snapshot)
        adapter.finish_import(exp, source="smac", filename=directory.name)
        self._report(exp)

    def _report(self, exp) -> None:
        self.stdout.write(self.style.SUCCESS(f"Imported experiment {exp.title!r} (id {exp.pk})"))
