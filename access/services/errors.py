"""`ServiceError` carries a translated message that a view shows as an error
and a management command raises as a `CommandError`."""


class ServiceError(Exception):
    """`ServiceError` means the request was understood and cannot be carried
    out; its message says why, in words fit for the person who asked."""
