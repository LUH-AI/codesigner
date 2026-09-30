from django.apps import AppConfig


class AccessConfig(AppConfig):
    name = "access"
    verbose_name = "Access control"

    def ready(self):
        from . import signals  # noqa: F401 — connects the receivers
