"""`llm` calls the OpenAI chat API synchronously, since turns run in a worker
thread (`.turns`).

Parts come from A3S's `a3s/agent/llm.py` (https://github.com/LUH-AI/A3S),
BSD-3-Clause, Copyright (c) Marcel Wever.
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

# Gemma models write calls as <|tool_call>call:NAME{KEY:<|"|>VALUE<|"|>, ...}
_TEXT_TOOL_CALL = re.compile(r"<\|tool_call\>call:(\w+)\{", re.DOTALL)
_TEXT_KV = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL)


@dataclass
class ToolCall:

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Reply:

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: The wire-format history entry.
    message: dict = field(default_factory=dict)
    #: Server-counted prompt plus completion tokens.
    tokens: int = 0


class Unavailable(Exception):
    """The model server failed."""


#: ("text", chunk) or ("thinking", True).
Progress = Callable[[str, object], None]


class ChatClient:

    def __init__(self, *, base_url: str, model: str, api_key: str, timeout: float = 120.0,
                 retries: int = 2, extra_body: dict | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                              timeout=timeout, max_retries=retries)
        self.model = model
        self._extra = dict(extra_body or {})

    @classmethod
    def from_settings(cls) -> "ChatClient":
        from django.conf import settings

        return cls(base_url=settings.ASSISTANT_LLM_BASE_URL, model=settings.ASSISTANT_LLM_MODEL,
                   api_key=settings.ASSISTANT_LLM_TOKEN, timeout=settings.ASSISTANT_LLM_TIMEOUT,
                   retries=settings.ASSISTANT_LLM_RETRIES,
                   extra_body=settings.ASSISTANT_LLM_EXTRA_BODY)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             progress: Progress | None = None, *, response_format: dict | None = None,
             extra_body: dict | None = None) -> Reply:
        """`chat` streams to *progress* when one is given. *extra_body* only
        extends the current request, deep-merged into the client's."""
        import openai

        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format
        extra = _merged(self._extra, extra_body or {})
        if extra:
            kwargs["extra_body"] = extra
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
                # Rescanning all text catches tags split across chunks.
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


def _merged(base: dict, over: dict) -> dict:
    """`_merged` deep-merges *over* onto *base*."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = value
    return out


def _held_back(visible: str) -> str:
    """`_held_back` trims partial `<think>`."""
    for n in range(min(len("<think>") - 1, len(visible)), 0, -1):
        if "<think>".startswith(visible[-n:]):
            return visible[:-n]
    return visible


def _reply(content: str | None, raw_calls: list[dict], tokens: int) -> Reply:
    """`_reply` falls back to calls written in the text."""
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


class EmbeddingClient:
    """`EmbeddingClient` embeds library text."""

    BATCH = 64

    def __init__(self, *, base_url: str, model: str, api_key: str, timeout: float = 120.0,
                 retries: int = 2) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                              timeout=timeout, max_retries=retries)
        self.model = model

    @classmethod
    def from_settings(cls) -> "EmbeddingClient | None":
        """`ASSISTANT_EMBED_BASE_URL` enables embedding."""
        from django.conf import settings

        if not settings.ASSISTANT_EMBED_BASE_URL:
            return None
        return cls(base_url=settings.ASSISTANT_EMBED_BASE_URL, model=settings.ASSISTANT_EMBED_MODEL,
                   api_key=settings.ASSISTANT_EMBED_TOKEN, timeout=settings.ASSISTANT_LLM_TIMEOUT,
                   retries=settings.ASSISTANT_LLM_RETRIES)

    def embed(self, texts: list[str]):
        """`embed` returns unit-length float32 rows."""
        import numpy as np
        import openai

        rows = []
        try:
            for start in range(0, len(texts), self.BATCH):
                response = self._client.embeddings.create(model=self.model,
                                                          input=texts[start:start + self.BATCH])
                rows += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            raise Unavailable(str(exc)) from exc
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1.0)
