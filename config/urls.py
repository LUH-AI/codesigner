"""Root URLconf: every incoming request is matched against these patterns.

Per-page routes live in web/urls.py; this file only mounts them (plus the
admin).  `include()` keeps app routes self-contained so apps stay portable.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("web.urls")),
]
