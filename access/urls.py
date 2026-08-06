"""Signing in and out.

Mounted unconditionally, like the middleware, so the two modes differ by one
boolean and not by which routes exist. With `REQUIRE_LOGIN` off these pages are
reachable but pointless — nothing sends you to them.

Only login and logout. Password reset needs a configured mail server, which is
an operator decision and a whole surface of its own; until someone asks for it,
accounts are created and passwords set in the Django admin.
"""

from django.contrib.auth import views as auth_views
from django.urls import path

app_name = "access"

urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="access/login.html"),
         name="login"),
    # Django's LogoutView is POST-only, which is right: a GET logout can be
    # triggered by any page that can make the browser fetch a URL.
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
