"""Views show a `ServiceError` message; commands re-raise it as `CommandError`."""


class ServiceError(Exception):
    """`ServiceError` carries a translated message for the person who asked."""
