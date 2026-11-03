"""`run_turn` runs one conversation turn: the model calls tools within limits
until it replies in words or asks the person something.

`emit` writes each page update as an `AssistantEvent`, so a reloaded page can
resume a turn running elsewhere.

`run_turn` adapts the loop of A3S (https://github.com/LUH-AI/A3S, BSD-3-Clause,
Copyright (c) Marcel Wever).
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

#: `HISTORY` caps the messages sent per call; `.Conversation.plan` carries what
#: matters from older ones.
HISTORY = 60
#: Streamed-text flush interval.
FLUSH_SECONDS = 0.15
#: `REPEATS` identical calls in one turn end the turn.
REPEATS = 3


def emit(conversation, kind: str, **payload) -> AssistantEvent:
    return AssistantEvent.objects.create(conversation=conversation, kind=kind, payload=payload)


def tokens_today(owner_id) -> int:
    """`tokens_today` takes None for an instance without accounts."""
    since = timezone.now() - timedelta(days=1)
    spent = (Message.objects.filter(conversation__owner_id=owner_id, created_at__gte=since)
             .aggregate(total=Sum("tokens"))["total"])
    return int(spent or 0)


def history(conversation) -> list[dict]:
    """`history` returns about the last `HISTORY` messages, starting at one of
    the person's, so every tool answer keeps its call."""
    messages = list(conversation.messages.all())
    start = max(0, len(messages) - HISTORY)
    while start > 0 and messages[start].role != Message.USER:
        start -= 1
    return [m.as_openai() for m in messages[start:]]


class _Stream:
    """`_Stream` batches streamed text."""

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
    """`run_turn` returns "answered", "waiting", or why it stopped:
    "unavailable", "budget", "steps", "tool_calls" or "repeating"."""
    from .gateway import Context
    from .llm import Unavailable
    from .prompts import system_prompt

    Message.objects.create(conversation=conversation, role=Message.USER, content=text)
    emit(conversation, "user", text=text)
    context = Context(conversation)
    calls_made, seen = 0, {}
    corrected = False

    steps = max(1, settings.ASSISTANT_MAX_STEPS)
    for step in range(steps):
        if tokens_today(conversation.owner_id) >= settings.ASSISTANT_DAILY_TOKENS:
            return _end(conversation, "budget")
        emit(conversation, "status", state="working")
        progress = _Stream(conversation) if stream else None
        messages = [{"role": "system", "content": system_prompt(conversation)}]
        messages += history(conversation)
        try:
            # The last step offers no tools, so every turn ends in an answer.
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
            invented = _invented(reply.content)
            if invented and not corrected and step < steps - 1:
                # Once per turn, so inventing still ends.
                corrected = True
                _warn_invented(conversation, invented)
                continue
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


def _invented(text: str) -> list[str]:
    from .library import uncited_keys

    return uncited_keys(text)


def _warn_invented(conversation, keys: list[str]) -> None:
    """`_warn_invented` marks the answer withdrawn and tells the model which
    *keys* name no library paper. The warning is stored with the user role for
    the model; no event shows it to the person."""
    named = ", ".join(f"[{key}]" for key in keys)
    emit(conversation, "withdrawn", keys=keys)
    Message.objects.create(conversation=conversation, role=Message.USER, content=(
        f"Codesigner's citation check, not the person: your answer cited {named}, which "
        "is not a paper in the library. The person cannot follow it, and it may not exist; "
        "they have been told your answer is withdrawn and a corrected one follows. Answer "
        "again in full. Cite a paper only by a key that search_knowledge returned, for what "
        "its passage says: search for what you meant to cite, and if the library has "
        "nothing on it, make the point without a citation and say the library has nothing."))


def _cite(conversation, text: str) -> None:
    """`_cite` lists cited papers under the answer."""
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
