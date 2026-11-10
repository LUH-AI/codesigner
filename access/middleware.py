"""`LoginWallMiddleware` extends Django's default-closed
`LoginRequiredMiddleware`, so a forgotten route stays protected, and passes
everything through while `REQUIRE_LOGIN` is off. One boolean separates hosted
and local installs, instead of a second settings module. `AccountGateMiddleware`
sends a signed-in person to the agreement and password pages until both are
done.
"""

from urllib.parse import urlencode

from django.conf import settings
from django.contrib.auth.middleware import LoginRequiredMiddleware  # pyright: ignore[reportAttributeAccessIssue]
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import redirect
from django.urls import reverse


class LoginWallMiddleware(LoginRequiredMiddleware):
    """`LoginWallMiddleware` reads `REQUIRE_LOGIN` on every request, so a test
    that switches the setting sees the change at once.
    """

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not settings.REQUIRE_LOGIN:
            return None
        return super().process_view(request, view_func, view_args, view_kwargs)


class AccountGateMiddleware:
    """`AccountGateMiddleware` sends a signed-in person to accept a new
    agreement, then to change their password when `manage.py users
    require-new-password` asked for it, before any other page. A page request is redirected, an htmx
    request gets `HX-Redirect`, and any other request gets 403."""

    #: Views a person needs to get through the gate.
    OPEN = {"ui:agreement", "ui:password_change", "ui:password_changed", "ui:logout",
            "set_language", "ui:healthz", "ui:account_setup", "ui:first_password",
            "ui:password_forgot", "ui:imprint", "ui:privacy", "ui:terms"}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        from access.services.passwords import agreement_due, must_change_password

        user = getattr(request, "user", None)
        if not settings.REQUIRE_LOGIN or user is None or not user.is_authenticated:
            return None
        match = request.resolver_match
        if match is None or match.view_name in self.OPEN:
            return None
        if agreement_due(user):
            target = reverse("ui:agreement")
        elif must_change_password(user):
            target = reverse("ui:password_change")
        else:
            return None
        if request.headers.get("HX-Request"):
            response = HttpResponse(status=204)
            response["HX-Redirect"] = target
            return response
        if request.method == "GET" and "text/event-stream" not in request.headers.get("Accept", ""):
            return redirect(f"{target}?{urlencode({'next': request.get_full_path()})}")
        return HttpResponseForbidden()
