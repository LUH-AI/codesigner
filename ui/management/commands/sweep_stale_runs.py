from django.core.management.base import BaseCommand

from ui.services.run import sweep_stale_runs


class Command(BaseCommand):
    help = "Mark runs left pending/running by a previous process as errored (run at startup)."

    def handle(self, *args, **options):
        n = sweep_stale_runs()
        self.stdout.write(self.style.SUCCESS(f"Swept {n} stale run(s)."))
