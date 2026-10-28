from django.apps import AppConfig


class UiConfig(AppConfig):
    name = "ui"
    # `sweep_stale_runs` clears stale runs at startup, since a query in
    # `ready()` would hit the dev database under tests.

    def ready(self):
        # A system check reports a missing or incomplete figure layout file.
        from django.core.checks import register

        from .layout import check
        register(check)
