"""Check what the configured model server can do for the assistant.

    manage.py assistant_probe
    manage.py assistant_probe --model DeepSeek-V4-Flash --thinking-off '{"chat_template_kwargs": {"enable_thinking": false}}'

Each check sends a short made-up request, never anything of a person's, and
reports what came back and how long it took: the model's status, a plain
reply, a tool round trip, several tool calls in one reply, streaming with token
counts, a reply in a JSON schema, how reasoning is returned, the switch that
turns reasoning off, and embeddings.
"""

import json
import time
import urllib.request

from django.conf import settings
from django.core.management.base import BaseCommand

ADD = {"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"},
                                                    "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}
WEATHER = {"type": "function", "function": {
    "name": "weather", "description": "The weather in one city today.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
ROUTE = {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["kind", "certainty"],
    "properties": {"kind": {"type": "string", "enum": ["greeting", "question", "request"]},
                   "certainty": {"type": "number"}}}}}


class Command(BaseCommand):
    help = "Check what the configured model server can do for the assistant."

    def add_arguments(self, parser):
        parser.add_argument("--model", help="A model other than ASSISTANT_LLM_MODEL.")
        parser.add_argument("--thinking-off", default='{"chat_template_kwargs": '
                                                      '{"enable_thinking": false}}',
                            help="The extra fields that should turn reasoning off, as JSON.")
        parser.add_argument("--skip-embeddings", action="store_true")

    def handle(self, *args, **options):
        from openai import OpenAI

        self.model = options["model"] or settings.ASSISTANT_LLM_MODEL
        self.extra = dict(settings.ASSISTANT_LLM_EXTRA_BODY)
        self.client = OpenAI(base_url=settings.ASSISTANT_LLM_BASE_URL,
                             api_key=settings.ASSISTANT_LLM_TOKEN or "not-needed",
                             timeout=settings.ASSISTANT_LLM_TIMEOUT, max_retries=0)
        self.stdout.write(f"{settings.ASSISTANT_LLM_BASE_URL}  model {self.model}\n")
        checks = [("listed", self.listed), ("plain reply", self.plain),
                  ("tool round trip", self.tool_round_trip),
                  ("parallel tool calls", self.parallel_calls),
                  ("streaming with usage", self.streaming),
                  ("JSON schema reply", self.json_schema),
                  ("reasoning returned as", self.reasoning),
                  ("reasoning switched off", lambda: self.reasoning_off(options["thinking_off"]))]
        if not options["skip_embeddings"]:
            checks.append(("embeddings", self.embeddings))
        for name, check in checks:
            started = time.monotonic()
            try:
                ok, detail = check()
            except Exception as exc:  # noqa: BLE001 — a failed check is reported and the rest run
                ok, detail = False, f"{type(exc).__name__}: {str(exc)[:300]}"
            seconds = time.monotonic() - started
            mark = self.style.SUCCESS("ok  ") if ok else self.style.ERROR("FAIL")
            self.stdout.write(f"{mark} {name:<24} {seconds:6.1f}s  {detail}")

    def _create(self, **kwargs):
        extra = {**self.extra, **kwargs.pop("extra_body", {})}
        if extra:
            kwargs["extra_body"] = extra
        return self.client.chat.completions.create(model=self.model, **kwargs)

    def listed(self):
        """`listed` reads the model's status from chat-ai's `/modellist`, or
        from `/models` on other servers."""
        base = settings.ASSISTANT_LLM_BASE_URL.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        request = urllib.request.Request(root + "/modellist",
                                         headers={"Authorization": f"Bearer {settings.ASSISTANT_LLM_TOKEN}"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                models = json.load(response)
        except Exception:  # noqa: BLE001 — servers without /modellist answer /models
            models = [{"id": m.id, "status": "listed"} for m in self.client.models.list()]
        found = {m["id"]: m.get("status", "?") for m in models}
        if self.model not in found:
            return False, f"not among {sorted(found)}"
        return found[self.model] in ("ready", "listed"), f"status {found[self.model]}"

    def plain(self):
        reply = self._create(messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        text = reply.choices[0].message.content or ""
        return "ready" in text.lower(), f"{text.strip()[:60]!r}, {_tokens(reply)} tokens"

    def tool_round_trip(self):
        messages = [{"role": "user", "content": "What is 17 + 25? Use the add tool."}]
        first = self._create(messages=messages, tools=[ADD], tool_choice="auto")
        message = first.choices[0].message
        if not message.tool_calls:
            return False, f"no tool call; said {(message.content or '')[:80]!r}"
        call = message.tool_calls[0]
        arguments = json.loads(call.function.arguments or "{}")
        messages.append({"role": "assistant", "content": message.content,
                         "tool_calls": [{"id": call.id, "type": "function",
                                         "function": {"name": call.function.name,
                                                      "arguments": call.function.arguments}}]})
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": str(arguments.get("a", 0) + arguments.get("b", 0))})
        second = self._create(messages=messages, tools=[ADD], tool_choice="auto")
        answer = second.choices[0].message.content or ""
        return "42" in answer, f"called add{arguments}, answered {answer.strip()[:60]!r}"

    def parallel_calls(self):
        reply = self._create(messages=[{"role": "user", "content":
                                        "What is the weather in Hannover and in Göttingen? "
                                        "Call the weather tool for both cities at once."}],
                             tools=[WEATHER], tool_choice="auto")
        calls = reply.choices[0].message.tool_calls or []
        cities = [json.loads(c.function.arguments or "{}").get("city") for c in calls]
        return len(calls) >= 2, f"{len(calls)} call(s) in one reply: {cities}"

    def streaming(self):
        started = time.monotonic()
        stream = self._create(messages=[{"role": "user", "content": "Count from one to five in words."}],
                              stream=True, stream_options={"include_usage": True})
        first, text, usage = None, "", None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.monotonic() - started
                text += chunk.choices[0].delta.content
        tokens = getattr(usage, "total_tokens", 0) if usage else 0
        first_text = f"{first:.1f}s" if first is not None else "never"
        return bool(tokens), f"first text after {first_text}, usage {tokens or 'missing'}"

    def json_schema(self):
        reply = self._create(messages=[{"role": "user", "content":
                                        "Classify this message: 'Hi there!'"}],
                             response_format=ROUTE)
        text = reply.choices[0].message.content or ""
        parsed = json.loads(text)
        ok = parsed.get("kind") in ("greeting", "question", "request") and set(parsed) == {"kind", "certainty"}
        return ok, " ".join(text.split())[:80]

    def reasoning(self):
        reply = self._create(messages=[{"role": "user", "content": "Is 91 prime? Answer yes or no."}])
        message = reply.choices[0].message
        separate = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if separate:
            return True, f"a separate field, {len(separate)} characters"
        if "<think>" in (message.content or ""):
            return True, "<think> tags in the text"
        return True, f"none visible, {_tokens(reply, 'completion')} completion tokens"

    def reasoning_off(self, switch):
        off = json.loads(switch)
        on = self._create(messages=[{"role": "user", "content": "Is 91 prime? Answer yes or no."}])
        quiet = self._create(messages=[{"role": "user", "content": "Is 91 prime? Answer yes or no."}],
                             extra_body=off)
        message = quiet.choices[0].message
        thought = (getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                   or "<think>" in (message.content or ""))
        before, after = _tokens(on, "completion"), _tokens(quiet, "completion")
        return not thought, f"{switch}: completion tokens {before} → {after}"

    def embeddings(self):
        from openai import OpenAI

        if not settings.ASSISTANT_EMBED_BASE_URL:
            return False, "ASSISTANT_EMBED_BASE_URL is not set"
        client = OpenAI(base_url=settings.ASSISTANT_EMBED_BASE_URL,
                        api_key=settings.ASSISTANT_EMBED_TOKEN or "not-needed", max_retries=0)
        texts = ["Bayesian optimization with a random forest surrogate",
                 "SMAC models the objective with a random forest",
                 "The recipe needs two eggs"]
        response = client.embeddings.create(model=settings.ASSISTANT_EMBED_MODEL, input=texts)
        import numpy as np

        vectors = np.asarray([d.embedding for d in sorted(response.data, key=lambda d: d.index)])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        near, far = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
        return near > far, (f"{settings.ASSISTANT_EMBED_MODEL}, {vectors.shape[1]} dimensions, "
                            f"related {near:.2f} vs unrelated {far:.2f}")


def _tokens(reply, kind="total"):
    usage = getattr(reply, "usage", None)
    return getattr(usage, f"{kind}_tokens", 0) if usage else 0
