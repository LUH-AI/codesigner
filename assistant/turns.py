"""Starting a turn without holding the request open for it.

A turn spends nearly all its time waiting on the model, seconds to a minute,
so it runs in a thread of this web process (`ASSISTANT_TURN_THREADS` at most)
rather than on the task queue, whose workers runs hold for hours. The page
follows it through its events (assistant/views.py), so nothing waits on the
thread itself.

One turn at a time per conversation: a second message while one is running is
refused rather than interleaved. A turn left behind by a server that stopped
mid-way is taken as over after `STALE`.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections, connections
from django.db.models import Q
from django.utils import timezone

from .models import Conversation

log = logging.getLogger(__name__)

#: How long a turn may claim its conversation before it is taken as abandoned.
STALE = timedelta(minutes=15)

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=max(1, settings.ASSISTANT_TURN_THREADS),
                                       thread_name_prefix="assistant-turn")
        return _pool


def busy(conversation) -> bool:
    """Whether a turn is running in *conversation*."""
    started = conversation.turn_started_at
    return started is not None and started > timezone.now() - STALE


def start(conversation, text: str) -> bool:
    """Begin a turn answering *text*. False if one is already running."""
    now = timezone.now()
    claimed = (Conversation.objects.filter(pk=conversation.pk)
               .filter(Q(turn_started_at__isnull=True) | Q(turn_started_at__lt=now - STALE))
               .update(turn_started_at=now))
    if not claimed:
        return False
    if settings.ASSISTANT_TURNS_INLINE:
        _run(conversation.pk, text)
    else:
        _executor().submit(_run, conversation.pk, text)
    return True


def tools_for(conversation):
    """The tools *conversation* is offered: every conversation's, those that act
    on the person's page, and while it plans an experiment the planning ones —
    after that, the lookups among them, without the plan's own, and those
    that read the experiment."""
    from .tools.basic import basic
    from .tools.client import client
    from .tools.data import data_tools
    from .tools.experiment import experiment_tools
    from .tools.knowledge import knowledge
    from .tools.planning import lookups, planning
    from .tools.registry import ToolRegistry

    planning_now = conversation.phase == Conversation.PLANNING
    # While planning there is no experiment to open or fill: only looking at
    # the page and pointing at what is on it.
    page = [t for t in client() if not planning_now or t.name in ("look_at_page", "highlight")]
    specs = basic() + data_tools() + knowledge() + page
    specs += planning() if planning_now else lookups() + experiment_tools()
    return ToolRegistry(specs)


def _run(conversation_id: int, text: str) -> None:
    from .llm import ChatClient
    from .loop import emit, run_turn

    close_old_connections()
    conversation = Conversation.objects.get(pk=conversation_id)
    try:
        run_turn(conversation, text, llm=ChatClient.from_settings(),
                 tools=tools_for(conversation))
    except Exception as exc:  # noqa: BLE001 — the page has to hear that the turn is over
        log.exception("assistant turn failed")
        emit(conversation, "turn_end", reason="failed", detail=str(exc)[:500])
    finally:
        Conversation.objects.filter(pk=conversation_id).update(turn_started_at=None)
        if not settings.ASSISTANT_TURNS_INLINE:
            connections.close_all()
