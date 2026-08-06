"""The login wall.

Django ships `LoginRequiredMiddleware`, which gates every view unless it is
marked with `login_not_required`. That is exactly the shape wanted here — the
default is *closed*, so a route added later is protected by having been
forgotten about, rather than exposed by it. The one thing it does not do is turn
itself off, which is what this subclass adds.

Installed unconditionally. With `REQUIRE_LOGIN` off it passes every request
straight through, so a local install has no accounts, no login page in its way,
and no behaviour to remember. That is the point of a switch rather than a
different settings module: the hosted and unhosted configurations are the same
configuration, and the difference is one boolean rather than a file that drifts.
"""

from django.conf import settings
from django.contrib.auth.middleware import LoginRequiredMiddleware


class LoginWallMiddleware(LoginRequiredMiddleware):
    """Require a signed-in user for everything, when `REQUIRE_LOGIN` is on.

    The setting is read per request rather than at startup. It costs nothing,
    and it means the wall can be switched in a test — or by an operator's
    `.env` — without the reading being stale.
    """

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not settings.REQUIRE_LOGIN:
            return None
        return super().process_view(request, view_func, view_args, view_kwargs)
