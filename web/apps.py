from django.apps import AppConfig


class WebConfig(AppConfig):
    name = "web"
    # Stale runs left by a restart are cleared by the `sweep_stale_runs`
    # management command, run at startup (see Step 10's container entrypoint).
    # We deliberately do NOT sweep in ready(): querying the database during app
    # initialization is discouraged and, under the test runner, would touch the
    # development database before the test database exists.
