"""Which conversation a page's chat shows, when the assistant is on."""

from django.conf import settings


def assistant(request):
    """`assistant` for the chat (assistant/_chat.html): the conversation this
    page continues — its experiment's on an experiment's pages, else the
    person's latest — and where its replay starts. Nothing when the assistant
    is off, or for somebody not signed in where accounts are on."""
    if not settings.ASSISTANT_ENABLED:
        return {}
    user = getattr(request, "user", None)
    if settings.REQUIRE_LOGIN and not getattr(user, "is_authenticated", False):
        return {}
    from .views import chat_context

    experiment_id = None
    match = getattr(request, "resolver_match", None)
    if match is not None and getattr(match.func, "experiment_action", None):
        experiment_id = match.kwargs.get("pk")
    url_name = match.view_name if match is not None else ""
    return {"assistant": chat_context(request, experiment_id, url_name)}
