"""The language model, over the OpenAI chat API.

Callers work with `Reply` and `ToolCall` rather than the SDK's types, so the
backend can change without touching them. Synchronous, because the assistant's
turns run in a worker thread beside the ORM (assistant/loop.py).

Ported from A3S's `a3s/agent/llm.py` (https://github.com/LUH-AI/A3S),
BSD-3-Clause, Copyright (c) Marcel Wever: the normalised reply, the assembly of
streamed tool-call fragments, the removal of `<think>` blocks, and the reading of
tool calls some local models write as text.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Some local models (gemma, for one) write a tool call into their text instead
# of calling it:  <|tool_call>call:NAME{KEY:<|"|>VALUE<|"|>, ...}
_TEXT_TOOL_CALL = re.compile(r"<\|tool_call\>call:(\w+)\{", re.DOTALL)
_TEXT_KV = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL)


@dataclass
class ToolCall:
    """One tool call the model asked for, its arguments already parsed."""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Reply:
    """What the model answered: text, tool calls, or both."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: The reply as an assistant message for the history, in the wire format.
    message: dict = field(default_factory=dict)
    #: Tokens the call cost, prompt and completion, as the server counted them.
    tokens: int = 0


class Unavailable(Exception):
    """The model could not be reached, or did not answer in time."""


#: Called while a reply streams in: `(kind, value)` with kind "text" and the
#: new text, or "thinking" and True while the model reasons before answering.
Progress = Callable[[str, object], None]


class ChatClient:
    """A model behind an OpenAI-compatible endpoint."""

    def __init__(self, *, base_url: str, model: str, api_key: str, timeout: float = 120.0,
                 retries: int = 2, repetition_penalty: float | None = None,
                 enable_thinking: bool | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                              timeout=timeout, max_retries=retries)
        self.model = model
        self._extra = {}
        if repetition_penalty is not None:
            self._extra["repetition_penalty"] = repetition_penalty
        if enable_thinking is not None:
            self._extra["enable_thinking"] = enable_thinking

    @classmethod
    def from_settings(cls) -> "ChatClient":
        from django.conf import settings

        return cls(base_url=settings.ASSISTANT_LLM_BASE_URL, model=settings.ASSISTANT_LLM_MODEL,
                   api_key=settings.ASSISTANT_LLM_TOKEN, timeout=settings.ASSISTANT_LLM_TIMEOUT,
                   retries=settings.ASSISTANT_LLM_RETRIES,
                   repetition_penalty=settings.ASSISTANT_LLM_REPETITION_PENALTY,
                   enable_thinking=settings.ASSISTANT_LLM_ENABLE_THINKING)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             progress: Progress | None = None) -> Reply:
        """Ask the model to answer *messages*, offering it *tools*. Streams when
        *progress* is given, which then hears the text as it arrives."""
        import openai

        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self._extra:
            kwargs["extra_body"] = dict(self._extra)
        try:
            if progress is None:
                return self._blocking(kwargs)
            return self._streaming(kwargs, progress)
        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            raise Unavailable(str(exc)) from exc

    def _blocking(self, kwargs: dict) -> Reply:
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content
        if content and "<think>" in content:
            content = _THINK.sub("", content).strip()
        raw_calls = [{"id": call.id, "type": "function",
                      "function": {"name": call.function.name,
                                   "arguments": call.function.arguments}}
                     for call in (message.tool_calls or [])]
        tokens = getattr(response.usage, "total_tokens", 0) or 0
        return _reply(content, raw_calls, tokens)

    def _streaming(self, kwargs: dict, progress: Progress) -> Reply:
        stream = self._client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        text, builders, tokens, shown = "", {}, 0, 0
        for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    tokens = getattr(chunk.usage, "total_tokens", 0) or tokens
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                progress("thinking", True)
            if delta.content:
                text += delta.content
                # What is outside the reasoning so far, recomputed from the whole
                # text so a tag split across chunks is still recognised, less an
                # ending that may yet turn out to open one; only what has not
                # been passed on yet is.
                visible = _held_back(_THINK.sub("", text).split("<think>", 1)[0])
                if len(visible) > shown:
                    progress("text", visible[shown:])
                    shown = len(visible)
                elif "<think" in text:
                    progress("thinking", True)
            for fragment in delta.tool_calls or []:
                built = builders.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                if fragment.id:
                    built["id"] = fragment.id
                if fragment.function:
                    if fragment.function.name:
                        built["name"] = fragment.function.name
                    if fragment.function.arguments:
                        built["arguments"] += fragment.function.arguments
        rest = _THINK.sub("", text).split("<think>", 1)[0]
        if len(rest) > shown:
            progress("text", rest[shown:])
        if "<think>" in text:
            text = _THINK.sub("", text).strip()
        raw_calls = [{"id": built["id"] or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                      "function": {"name": built["name"], "arguments": built["arguments"]}}
                     for _, built in sorted(builders.items())]
        return _reply(text or None, raw_calls, tokens)


def _held_back(visible: str) -> str:
    """*visible* without an ending that could be the start of `<think>`."""
    for n in range(min(len("<think>") - 1, len(visible)), 0, -1):
        if "<think>".startswith(visible[-n:]):
            return visible[:-n]
    return visible


def _reply(content: str | None, raw_calls: list[dict], tokens: int) -> Reply:
    """A reply from the model's text and its calls in the wire format, reading a
    call some models write into their text when they made none properly."""
    calls = []
    for raw in raw_calls:
        try:
            arguments = json.loads(raw["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            log.warning("unparseable arguments for %s: %r", raw["function"]["name"],
                        raw["function"]["arguments"])
            arguments = {"__unparseable__": raw["function"]["arguments"]}
        calls.append(ToolCall(id=raw["id"], name=raw["function"]["name"],
                              arguments=arguments if isinstance(arguments, dict) else {}))
    if not calls and content and "<|tool_call>" in content:
        calls = _text_tool_calls(content)
        raw_calls = [{"id": call.id, "type": "function",
                      "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                     for call in calls]
        if calls:
            content = None
    message: dict = {"role": "assistant", "content": content or ""}
    if raw_calls:
        message = {"role": "assistant", "content": content or None, "tool_calls": raw_calls}
    return Reply(content=content, tool_calls=calls, message=message, tokens=tokens)


def _text_tool_calls(content: str) -> list[ToolCall]:
    """Tool calls written into the text in the gemma style."""
    calls = []
    for match in _TEXT_TOOL_CALL.finditer(content):
        rest = content[match.end():]
        arguments = {kv.group(1): kv.group(2) for kv in _TEXT_KV.finditer(rest)}
        if not arguments:
            close = rest.find("}")
            if close > 0:
                with contextlib.suppress(json.JSONDecodeError):
                    arguments = json.loads("{" + rest[:close] + "}")
        calls.append(ToolCall(id=f"parsed_{uuid.uuid4().hex[:8]}", name=match.group(1),
                              arguments=arguments))
    return calls
