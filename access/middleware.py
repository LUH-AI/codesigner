"""`LoginWallMiddleware` extends Django's default-closed
`LoginRequiredMiddleware`, so a forgotten route stays protected, and passes
everything through while `REQUIRE_LOGIN` is off. One boolean separates hosted
and local installs, instead of a second settings module.
"""

from django.conf import settings
from django.contrib.auth.middleware import LoginRequiredMiddleware  # pyright: ignore[reportAttributeAccessIssue]


class LoginWallMiddleware(LoginRequiredMiddleware):
    """`LoginWallMiddleware` reads `REQUIRE_LOGIN` on every request, so a test
    that switches the setting sees the change at once.
    """

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not settings.REQUIRE_LOGIN:
            return None
        return super().process_view(request, view_func, view_args, view_kwargs)
