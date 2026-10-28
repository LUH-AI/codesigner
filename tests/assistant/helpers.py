"""A model that answers from a script, for testing turns without a server.

A3S's `MockLLM` and response builders (https://github.com/LUH-AI/A3S,
BSD-3-Clause, Copyright (c) Marcel Wever), made synchronous like the client.
"""

from __future__ import annotations

import json

from assistant.llm import Reply, ToolCall


class MockLLM:
    """`MockLLM` answers each `chat` with the next queued reply and records
    every call. An empty queue makes `chat` say so in a text reply, which ends
    the turn."""

    model = "mock"

    def __init__(self, *replies):
        self.queue = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, progress=None):
        self.calls.append({"messages": messages, "tools": tools})
        reply = self.queue.pop(0) if self.queue else text("[no more replies]")
        if isinstance(reply, Exception):
            raise reply
        if progress is not None and reply.content:
            progress("text", reply.content)
        return reply


def text(content: str, tokens: int = 10) -> Reply:
    """`text` builds a reply in words."""
    return Reply(content=content, tool_calls=[],
                 message={"role": "assistant", "content": content}, tokens=tokens)


def call(name: str, tokens: int = 10, **arguments) -> Reply:
    """`call` builds a reply that calls one tool."""
    return calls((name, arguments), tokens=tokens)


def calls(*named, tokens: int = 10) -> Reply:
    """`calls` builds a reply that calls several tools, each given as
    `(name, arguments)`."""
    made, wire = [], []
    for index, (name, arguments) in enumerate(named):
        call_id = f"call_{index}_{name}"
        made.append(ToolCall(id=call_id, name=name, arguments=arguments))
        wire.append({"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": json.dumps(arguments)}})
    return Reply(content=None, tool_calls=made,
                 message={"role": "assistant", "content": None, "tool_calls": wire},
                 tokens=tokens)
