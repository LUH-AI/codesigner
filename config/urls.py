"""Root URLconf: every incoming request is matched against these patterns.

Per-page routes live in ui/urls.py; this file only mounts them (plus the
admin).  `include()` keeps app routes self-contained so apps stay portable.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.decorators import login_not_required
from django.urls import include, path
from django.views.i18n import set_language

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("access.urls")),
    # Django's set_language view, for the sidebar language switcher. Spelled out
    # rather than `include("django.conf.urls.i18n")` — which is the same single
    # route — so it can be exempted from the login wall: the login page carries
    # the switcher, and choosing a language you can read should not require
    # signing in first.
    path("i18n/setlang/", login_not_required(set_language), name="set_language"),
    path("", include("ui.urls")),
]

# Uploaded datasets and models, served straight from disk for convenience while
# developing. Never on an instance with accounts: MEDIA_ROOT is one flat
# directory, so this would hand every signed-in user every other user's data at
# a guessable URL. In the container nothing serves it at all (WhiteNoise handles
# STATIC_ROOT only).
if settings.DEBUG and not settings.REQUIRE_LOGIN:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
