"""Root URLconf: every incoming request is matched against these patterns.

Per-page routes live in web/urls.py; this file only mounts them (plus the
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
    path("", include("web.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
