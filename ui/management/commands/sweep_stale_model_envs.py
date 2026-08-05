from django.core.management.base import BaseCommand

from ui.services.modelenv import sweep_stale_environments


class Command(BaseCommand):
    help = "Mark model environments left 'preparing' by an interrupted worker as failed."

    def handle(self, *args, **options):
        count = sweep_stale_environments()
        self.stdout.write(self.style.SUCCESS(f"Swept {count} stale model environment(s)."))
