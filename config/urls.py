"""Root URLconf: every incoming request is matched against these patterns.

Per-page routes live in ui/urls.py; this file only mounts them (plus the
admin).  `include()` keeps app routes self-contained so apps stay portable.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    # Django's set_language view, for the sidebar language switcher.
    path("i18n/", include("django.conf.urls.i18n")),
    path("", include("ui.urls")),
]

# Uploaded datasets and models, served straight from disk for convenience while
# developing. Never on an instance with accounts: MEDIA_ROOT is one flat
# directory, so this would hand every signed-in user every other user's data at
# a guessable URL. In the container nothing serves it at all (WhiteNoise handles
# STATIC_ROOT only).
if settings.DEBUG and not settings.REQUIRE_LOGIN:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
