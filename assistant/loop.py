"""One turn of a conversation: the person says something, and the model answers
— calling tools as often as it needs to, within limits — until it replies in
words or asks the person something.

Everything a page should show is written as an `AssistantEvent` as it happens,
so a page can follow a turn running elsewhere and pick it up again after being
reloaded. Every message, tool calls and their answers included, is kept as the
model saw it, so the next turn continues the same conversation.

A3S ran one copy of this loop per phase (https://github.com/LUH-AI/A3S, BSD-3-
Clause, Copyright (c) Marcel Wever); this is one loop for every turn, keeping
from it the nudge-free handing back of control on a question, and the giving up
on a model that repeats itself.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

from django.conf import settings
from django.db.models import Sum
from django.utils import timezone

from .models import AssistantEvent, Message
from .tools.registry import ToolRegistry

#: How many messages of history are sent with each call. Older ones stay in the
#: conversation; the plan carries what matters of them.
HISTORY = 60
#: How often streamed text is written out, in seconds.
FLUSH_SECONDS = 0.15
#: The same call this many times in one turn is a model going round in circles.
REPEATS = 3


def emit(conversation, kind: str, **payload) -> AssistantEvent:
    """Write an event for pages showing *conversation*."""
    return AssistantEvent.objects.create(conversation=conversation, kind=kind, payload=payload)


def tokens_today(owner_id) -> int:
    """Model tokens spent in the last day by *owner_id*'s conversations (None:
    the conversations of an instance without accounts)."""
    since = timezone.now() - timedelta(days=1)
    spent = (Message.objects.filter(conversation__owner_id=owner_id, created_at__gte=since)
             .aggregate(total=Sum("tokens"))["total"])
    return int(spent or 0)


def history(conversation) -> list[dict]:
    """The conversation as the model is sent it: about its last `HISTORY`
    messages, reaching back to start at one of the person's, so no tool answer
    arrives without the call it answers and the message being answered is
    always there."""
    messages = list(conversation.messages.all())
    start = max(0, len(messages) - HISTORY)
    while start > 0 and messages[start].role != Message.USER:
        start -= 1
    return [m.as_openai() for m in messages[start:]]


class _Stream:
    """Streamed text, written out in pieces rather than per token."""

    def __init__(self, conversation):
        self.conversation = conversation
        self.pending = ""
        self.last = time.monotonic()
        self.thinking = False

    def __call__(self, kind, value):
        if kind == "thinking":
            if not self.thinking:
                self.thinking = True
                emit(self.conversation, "status", state="thinking")
            return
        self.thinking = False
        self.pending += value
        if time.monotonic() - self.last >= FLUSH_SECONDS:
            self.flush()

    def flush(self):
        if self.pending:
            emit(self.conversation, "text", text=self.pending)
            self.pending = ""
        self.last = time.monotonic()


def run_turn(conversation, text: str, *, llm, tools: ToolRegistry, stream: bool = True) -> str:
    """The person's *text*, and the model's answer to it. Returns how the turn
    ended: "answered", "waiting" (for the person's choice), or why it stopped —
    "unavailable", "budget", "steps", "tool_calls" or "repeating"."""
    from .gateway import Context
    from .llm import Unavailable
    from .prompts import system_prompt

    Message.objects.create(conversation=conversation, role=Message.USER, content=text)
    emit(conversation, "user", text=text)
    context = Context(conversation)
    calls_made, seen = 0, {}

    steps = max(1, settings.ASSISTANT_MAX_STEPS)
    for step in range(steps):
        if tokens_today(conversation.owner_id) >= settings.ASSISTANT_DAILY_TOKENS:
            return _end(conversation, "budget")
        emit(conversation, "status", state="working")
        progress = _Stream(conversation) if stream else None
        messages = [{"role": "system", "content": system_prompt(conversation)}]
        messages += history(conversation)
        try:
            # The last step offers no tools, so a turn always ends in an answer
            # to the person rather than in one more call.
            reply = llm.chat(messages, tools=tools.schemas if step < steps - 1 else None,
                             progress=progress)
        except Unavailable as exc:
            return _end(conversation, "unavailable", detail=str(exc))
        if progress is not None:
            progress.flush()
        elif reply.content:
            emit(conversation, "text", text=reply.content)
        Message.objects.create(conversation=conversation, role=Message.ASSISTANT,
                               content=reply.content or "",
                               tool_calls=reply.message.get("tool_calls"), tokens=reply.tokens)
        if not reply.tool_calls:
            _cite(conversation, reply.content)
            return _end(conversation, "answered")

        waiting, stopped = False, None
        for call in reply.tool_calls:
            if waiting or stopped:
                _answer(conversation, call, "Not done: the turn ended before this call.")
                continue
            calls_made += 1
            if calls_made > settings.ASSISTANT_MAX_TOOL_CALLS:
                stopped = "tool_calls"
                _answer(conversation, call, "Not done: too many tool calls in one turn.")
                continue
            signature = call.name + json.dumps(call.arguments, sort_keys=True, default=str)
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] >= REPEATS:
                stopped = "repeating"
                _answer(conversation, call, "Not done: this call was already made twice.")
                continue
            emit(conversation, "tool_call", tool=call.name, arguments=call.arguments)
            result = tools.execute(call.name, call.arguments, context)
            _answer(conversation, call, result.output)
            emit(conversation, "tool_result", tool=call.name, output=result.output[:3000])
            if result.waits_for_person:
                waiting = True
                emit(conversation, "choices", **result.show)
        if stopped:
            return _end(conversation, stopped)
        if waiting:
            return _end(conversation, "waiting")
    return _end(conversation, "steps")


def _cite(conversation, text: str) -> None:
    """The papers the answer cites from the library, listed under it."""
    from .library import cited

    sources = cited(text)
    if sources:
        emit(conversation, "sources", papers=sources)


def _answer(conversation, call, output: str) -> None:
    Message.objects.create(conversation=conversation, role=Message.TOOL,
                           tool_call_id=call.id, content=output)


def _end(conversation, reason: str, **detail) -> str:
    emit(conversation, "turn_end", reason=reason, **detail)
    conversation.save(update_fields=["updated_at"])
    return reason
