"""Signing in and out.

Mounted unconditionally, like the middleware, so the two modes differ by one
boolean and not by which routes exist. With `REQUIRE_LOGIN` off these pages are
reachable but pointless — nothing sends you to them.

Signing in, signing out, and changing the password you are signed in with.

Not password *reset*, which is the other thing: it mails a link to somebody who
cannot sign in, so it needs a configured mail server — an operator decision and a
whole surface of its own. Changing a password you already know needs none, which
is why one is here and the other is not. Until reset is asked for, a forgotten
password is reset by an administrator in the Django admin.

There is no registration either, by design. Nothing here accepts an
unauthenticated POST that creates an account: on an instance where uploading a
model is arbitrary code execution, an open door is the thing most worth not
having. Accounts are created by an administrator.
"""

from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

app_name = "access"

urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="access/login.html"),
         name="login"),
    # Django's LogoutView is POST-only, which is right: a GET logout can be
    # triggered by any page that can make the browser fetch a URL.
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    # Django's own views and its own form, which already enforces the password
    # validators in settings and asks for the current password before changing
    # it — so a borrowed, still-signed-in browser cannot lock its owner out.
    path("password/", auth_views.PasswordChangeView.as_view(
        template_name="access/password_change.html",
        success_url=reverse_lazy("access:password_changed")), name="password_change"),
    path("password/done/", auth_views.PasswordChangeDoneView.as_view(
        template_name="access/password_changed.html"), name="password_changed"),
]
