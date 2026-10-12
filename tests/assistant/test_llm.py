"""The model client: what it makes of a server's replies.

Against a stand-in for the OpenAI SDK's client, so the parsing — whole and
streamed replies, tool calls in pieces, reasoning to hide, tool calls written as
text — is checked without a server.
"""

from types import SimpleNamespace as NS

import httpx2
import openai
import pytest

from assistant.llm import ChatClient, Unavailable, _reply


def _client(create):
    client = ChatClient(base_url="http://model.invalid/v1", model="m", api_key="")
    client._client = NS(chat=NS(completions=NS(create=create)))
    return client


def _chunk(content=None, tool_calls=None, usage=None):
    if usage is not None:
        return NS(choices=[], usage=usage)
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls,
                                   reasoning_content=None))], usage=None)


def test_a_whole_reply_is_read_with_its_tool_calls_and_tokens():
    """What: a non-streamed reply's text, tool calls and token count.
    How: a stand-in response with one call."""
    response = NS(
        choices=[NS(message=NS(content="<think>hmm</think>Looking.", tool_calls=[
            NS(id="c1", function=NS(name="profile_dataset", arguments='{"demo": "iris"}'))]))],
        usage=NS(total_tokens=42))
    reply = _client(lambda **kw: response).chat([{"role": "user", "content": "hi"}])

    assert reply.content == "Looking."
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("profile_dataset", {"demo": "iris"})]
    assert reply.tokens == 42
    assert reply.message["tool_calls"][0]["id"] == "c1"


def test_a_streamed_reply_is_passed_on_without_its_reasoning():
    """What: streamed text reaches the listener as it arrives, without the
    model's reasoning, even when a tag is split across pieces; tool-call pieces
    are put back together.
    How: a stand-in stream of chunks."""
    stream = [
        _chunk("<thi"), _chunk("nk>weighing it</think>"), _chunk("Hello "), _chunk("there."),
        _chunk(tool_calls=[NS(index=0, id="c1", function=NS(name="explain_how", arguments='{"act'))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='ion": "run"}'))]),
        _chunk(usage=NS(total_tokens=99)),
    ]
    heard = []
    reply = _client(lambda **kw: iter(stream)).chat(
        [{"role": "user", "content": "hi"}], progress=lambda kind, value: heard.append((kind, value)))

    assert "".join(v for k, v in heard if k == "text") == "Hello there."
    assert ("thinking", True) in heard
    assert reply.content == "Hello there."
    assert reply.tool_calls[0].arguments == {"action": "run"}
    assert reply.tokens == 99


def test_an_unreachable_server_is_unavailable():
    """What: a connection that fails is `Unavailable`, which ends a turn quietly.
    How: a stand-in client raising the SDK's connection error."""
    def refuse(**kwargs):
        raise openai.APIConnectionError(request=httpx2.Request("POST", "http://model.invalid"))

    with pytest.raises(Unavailable):
        _client(refuse).chat([{"role": "user", "content": "hi"}])


def test_tool_calls_written_as_text_are_read_as_calls():
    """What: a model that writes its call into its text in the gemma style is
    understood as having called it.
    How: `_reply` on such text."""
    reply = _reply('<|tool_call>call:explain_how{action:<|"|>run<|"|>}', [], 0)

    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("explain_how", {"action": "run"})]
    assert reply.content is None


def test_arguments_that_are_not_json_are_kept_to_be_refused():
    """What: arguments the model garbled are not silently emptied — a tool would
    then run with none — but kept under a name no schema allows.
    How: `_reply` on a call with broken JSON."""
    raw = [{"id": "c", "type": "function", "function": {"name": "explain_how", "arguments": "{oops"}}]
    reply = _reply(None, raw, 0)

    assert reply.tool_calls[0].arguments == {"__unparseable__": "{oops"}


def test_server_specific_switches_are_sent_only_when_set():
    """What: repetition penalty and thinking, which some servers refuse, are only
    sent when configured.
    How: what the stand-in client is called with, with and without them."""
    seen = []
    response = NS(choices=[NS(message=NS(content="ok", tool_calls=None))], usage=None)

    def create(**kwargs):
        seen.append(kwargs)
        return response

    plain = _client(create)
    plain.chat([])
    tuned = ChatClient(base_url="http://model.invalid/v1", model="m", api_key="",
                       repetition_penalty=1.05, enable_thinking=False)
    tuned._client = NS(chat=NS(completions=NS(create=create)))
    tuned.chat([])

    assert "extra_body" not in seen[0]
    assert seen[1]["extra_body"] == {"repetition_penalty": 1.05, "enable_thinking": False}
