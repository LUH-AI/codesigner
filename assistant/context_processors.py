"""`assistant` picks the chats a page shows."""

from django.conf import settings


def assistant(request):
    """`assistant` returns the conversations the page's chats show, or an empty
    context when the assistant is off or `REQUIRE_LOGIN` turns a signed-out
    visitor away."""
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
