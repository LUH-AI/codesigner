"""`urlpatterns` mounts the admin and the routes in `ui/urls.py`.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.decorators import login_not_required  # pyright: ignore[reportAttributeAccessIssue]
from django.urls import include, path
from django.views.i18n import set_language

urlpatterns = [
    path("admin/", admin.site.urls),
    # `set_language` is spelled out so `login_not_required` can exempt it; the
    # login page carries the language switcher.
    path("i18n/setlang/", login_not_required(set_language), name="set_language"),
    path("", include("ui.urls")),
    path("", include("assistant.urls")),
]

# Uploads are served only in local development, since the flat `MEDIA_ROOT`
# would expose every user's data to every account.
if settings.DEBUG and not settings.REQUIRE_LOGIN:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
