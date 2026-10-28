"""`prune_model_envs` prunes uv's cache."""

import subprocess

from django.core.management.base import BaseCommand, CommandError

from experiments.services.modelenv import uv_path


class Command(BaseCommand):
    help = "Remove unused entries from uv's cache of built model environments."

    def handle(self, *args, **options):
        uv = uv_path()
        if not uv:
            raise CommandError("uv is not installed, so there is no cache to prune.")

        done = subprocess.run([uv, "cache", "prune"], capture_output=True, text=True,
                              check=False)
        if done.returncode != 0:
            raise CommandError((done.stderr or done.stdout).strip())
        self.stdout.write(done.stdout.strip() or "Nothing to prune.")
        # Experiment deletion leaves its uv environment behind.
        self.stdout.write(self.style.WARNING(
            "Environments belonging to deleted experiments are not tracked; "
            "this prunes whatever uv considers unused."))
