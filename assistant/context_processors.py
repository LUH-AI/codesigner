"""`assistant` picks the chats a page shows, when the assistant is on."""

from django.conf import settings


def assistant(request):
    """`assistant` gives the chats (assistant/_dock.html, assistant/_chat.html)
    the conversations the page shows (the general chat, its experiment's on an
    experiment's pages, and a visitor's) and where each replay starts. The
    context is empty when the assistant is off, for the instance or for the
    person, or for a signed-out person where accounts are on."""
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
