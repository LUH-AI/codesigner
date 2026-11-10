"""`gateway` lets a tool act as the conversation's owner, never with more
rights. Each tool call gets a fresh `AgentRequest`, and the policy denies that
request the `HUMAN_ONLY` actions. Tools only reach experiments through
`resolve_experiment`, the check views make.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import Http404

from access.policy import HUMAN_ONLY

from ui.permissions import BOTH, resolve_experiment

from .tools.registry import Refused


class AgentRequest:
    """`AgentRequest` is the request services and the policy receive when the
    assistant acts, with no browser behind it."""

    via_agent = True
    method = "POST"

    def __init__(self, user, session_key: str = "") -> None:
        self.user = user
        self.session_key = session_key


#: The assistant never uses the permissions `_escape_permissions` returns, even
#: for a person who has them.
def _escape_permissions():
    from access.policy import USE_CUSTOM_MODELS

    return frozenset({USE_CUSTOM_MODELS})


@dataclass
class Context:
    """`Context` gives a tool its conversation, and with it the person."""

    conversation: object

    def request(self) -> AgentRequest:
        """With `REQUIRE_LOGIN` off, `request` acts as an anonymous visitor.
        Otherwise it raises `Refused` unless the owner exists, is active and,
        for a browser conversation, is still signed in there."""
        if not settings.REQUIRE_LOGIN:
            from django.contrib.auth.models import AnonymousUser

            return AgentRequest(AnonymousUser(), self.conversation.session_key)
        from django.contrib.auth import get_user_model

        owner_id = self.conversation.owner_id
        user = get_user_model().objects.filter(pk=owner_id).first() if owner_id else None
        if user is None or not user.is_active:
            raise Refused("the person this conversation is for cannot be acted for")
        if self.conversation.session_key and not _signed_in(self.conversation.session_key, user):
            raise Refused("the person has signed out; nothing can be done for them until "
                          "they sign in again")
        escape = _escape_permissions()
        holds = user.has_perm

        def has_perm(perm, obj=None):
            return perm not in escape and holds(perm, obj)

        user.has_perm = has_perm
        return AgentRequest(user, self.conversation.session_key)

    def experiment(self, pk, action, drafts=BOTH):
        """`experiment` resolves *pk* for *action* as views do and raises
        `Refused` on failure."""
        request = self.request()
        if action in HUMAN_ONLY:
            raise Refused(f"only the person can {action} an experiment; explain how instead")
        try:
            return resolve_experiment(request, pk, action, drafts)
        except Http404:
            raise Refused(f"there is no experiment {pk} that the person can reach this way")
        except PermissionDenied:
            raise Refused(f"the person may not {action} experiment {pk}")


def _signed_in(session_key: str, user) -> bool:
    """`_signed_in` returns False after a sign-out or password change."""
    from importlib import import_module

    from django.contrib.auth import HASH_SESSION_KEY, SESSION_KEY
    from django.utils.crypto import constant_time_compare

    store = import_module(settings.SESSION_ENGINE).SessionStore(session_key=session_key)
    data = store.load()
    if str(data.get(SESSION_KEY)) != str(user.pk):
        return False
    return constant_time_compare(data.get(HASH_SESSION_KEY, ""), user.get_session_auth_hash())
