"""The one way a tool reaches the application: as the person, and never as
more than them.

Every tool call gets a request built here, afresh: the conversation's owner
loaded from the database again, refused if they can no longer be acted for, and
marked `via_agent`, which the policy narrows by `ui.permissions.HUMAN_ONLY`
whatever the person may do. An experiment is reached only through
`ui.permissions.resolve_experiment`, the check every view makes — so a tool can
reach what the person's own pages can, minus what they must do themselves.
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
    """What services and the policy are given as the request when the assistant
    acts: who for, and that it is the assistant. Nothing else of a request —
    there is no browser here, and no tool may read one."""

    via_agent = True
    method = "POST"

    def __init__(self, user, session_key: str = "") -> None:
        self.user = user
        self.session_key = session_key


#: Permissions that reach beyond a person's own work — every experiment, the
#: site, the instance's defaults, running uploaded code. A person who holds them
#: uses them themselves; the assistant acts as if they did not.
def _escape_permissions():
    from access.policy import CHANGE_DEFAULTS, MANAGE_ALL, MANAGE_SITE, USE_CUSTOM_MODELS, VIEW_ALL

    return frozenset({MANAGE_ALL, VIEW_ALL, MANAGE_SITE, CHANGE_DEFAULTS, USE_CUSTOM_MODELS})


@dataclass
class Context:
    """What a tool is given: the conversation it acts in, and through it, the
    person it acts for."""

    conversation: object

    def request(self) -> AgentRequest:
        """A request for the person, checked now: still there, still active,
        and — for a conversation started from a browser — still signed in there.
        Raises `Refused` when they cannot be acted for."""
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
        """Experiment *pk*, for the person to do *action* to, through the check
        every view makes. Raises `Refused` with why not."""
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
    """Whether the browser session *session_key* is signed in as *user*, and
    their password has not changed since."""
    from importlib import import_module

    from django.contrib.auth import HASH_SESSION_KEY, SESSION_KEY
    from django.utils.crypto import constant_time_compare

    store = import_module(settings.SESSION_ENGINE).SessionStore(session_key=session_key)
    data = store.load()
    if str(data.get(SESSION_KEY)) != str(user.pk):
        return False
    return constant_time_compare(data.get(HASH_SESSION_KEY, ""), user.get_session_auth_hash())
