"""`assistant` supplies page chats."""

from django.conf import settings


def assistant(request):
    """`assistant` returns {} when the assistant is off for the site or for
    the signed-in person, or when `REQUIRE_LOGIN` excludes the visitor."""
    if not settings.ASSISTANT_ENABLED:
        return {}
    user = getattr(request, "user", None)
    if settings.REQUIRE_LOGIN and not getattr(user, "is_authenticated", False):
        return {}
    from .views import chats_context, on_for

    if not on_for(request):
        return {}

    experiment_id = None
    match = getattr(request, "resolver_match", None)
    if match is not None and getattr(match.func, "experiment_action", None):
        experiment_id = match.kwargs.get("pk")
    url_name = match.view_name if match is not None else ""
    return {"assistant": chats_context(request, experiment_id, url_name)}
