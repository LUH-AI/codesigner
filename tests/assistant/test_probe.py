"""`manage.py assistant_probe` against a stand-in server: each check reports
what it found, and one that fails leaves the others to run."""

import json
from io import StringIO
from types import SimpleNamespace as NS

from django.core.management import call_command


class _Server:
    """`_Server` stands in for the OpenAI client: tool calls, schema replies and
    usage as a capable server gives them, except where `broken` says."""

    def __init__(self, broken=()):
        self.broken = set(broken)
        self.chat = NS(completions=NS(create=self.create))
        self.models = NS(list=lambda: [NS(id="m")])
        self.embeddings = NS(create=self.embed)

    def _call(self, name, arguments, index=0):
        return NS(id=f"c{index}", function=NS(name=name, arguments=json.dumps(arguments)))

    def create(self, model, messages, tools=None, stream=False, response_format=None, **kwargs):
        usage = NS(total_tokens=20, completion_tokens=5)
        if stream:
            return iter([NS(choices=[NS(delta=NS(content="one two"))], usage=None),
                         NS(choices=[], usage=usage)])
        last = messages[-1]["content"] or ""
        message = NS(content="", tool_calls=None, reasoning_content=None)
        if response_format:
            message.content = json.dumps({"kind": "greeting", "certainty": 0.9})
        elif tools and tools[0]["function"]["name"] == "weather":
            message.tool_calls = [self._call("weather", {"city": c}, i)
                                  for i, c in enumerate(["Hannover", "Göttingen"])]
        elif tools and messages[-1]["role"] == "tool":
            message.content = "It is 42."
        elif tools:
            message.tool_calls = (None if "tools" in self.broken
                                  else [self._call("add", {"a": 17, "b": 25})])
            message.content = "I would rather not." if "tools" in self.broken else None
        elif "prime" in last:
            message.content = "no"
            message.reasoning_content = None if kwargs.get("extra_body") else "7 × 13 = 91"
        else:
            message.content = "ready"
        return NS(choices=[NS(message=message)], usage=usage)

    def embed(self, model, input):
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        return NS(data=[NS(index=i, embedding=v) for i, v in enumerate(vectors)])


def _probe(monkeypatch, settings, server):
    import openai

    settings.ASSISTANT_LLM_MODEL = "m"
    settings.ASSISTANT_LLM_EXTRA_BODY = {}
    settings.ASSISTANT_EMBED_BASE_URL = "http://embed.invalid/v1"
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: server)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("no modellist")))
    out = StringIO()
    call_command("assistant_probe", stdout=out)
    return out.getvalue()


def test_a_capable_server_passes_every_check(monkeypatch, settings):
    """What: each of the probe's checks passes against a server that does
    everything. How: the stand-in answers tool calls, schema replies, streams
    with usage, separates reasoning and embeds; `/modellist` is unreachable,
    so the model is found through `/models`."""
    report = _probe(monkeypatch, settings, _Server())

    assert "FAIL" not in report
    assert "2 call(s) in one reply" in report
    assert "a separate field" in report


def test_a_failing_check_is_reported_and_the_rest_still_run(monkeypatch, settings):
    """What: a model that refuses tools fails the tool checks only. How: the
    stand-in answers tool requests in words; the later checks still report."""
    report = _probe(monkeypatch, settings, _Server(broken={"tools"}))

    lines = {line.split()[1]: line for line in report.splitlines()[1:]}
    assert "FAIL" in lines["tool"]
    assert "no tool call" in lines["tool"]
    assert "FAIL" not in lines["embeddings"]
