from django.apps import AppConfig


class UiConfig(AppConfig):
    name = "ui"
    # Stale runs left by a restart are cleared by the `sweep_stale_runs`
    # management command, run at startup (see Step 10's container entrypoint).
    # We deliberately do NOT sweep in ready(): querying the database during app
    # initialization is discouraged and, under the test runner, would touch the
    # development database before the test database exists.

    def ready(self):
        # The site's default figure layout is a required file; the app does not
        # start without a readable one that places every figure. A check rather
        # than an import-time error, so `manage.py check` reports it plainly.
        from django.core.checks import register

        from .layout import check
        register(check)
