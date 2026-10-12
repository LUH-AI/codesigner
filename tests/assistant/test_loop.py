"""A turn: the person speaks, the model answers, calling tools as it needs to.

Run against a scripted model (helpers.MockLLM), so every path — an answer, a
tool and its result, a question for the person, a refusal, and each way a turn
is cut short — is exercised without a server.
"""

import pytest

from assistant.llm import Unavailable
from assistant.loop import history, run_turn, tokens_today
from assistant.models import AgentAction, Conversation, Message
from assistant.tools.basic import basic
from assistant.tools.registry import ToolRegistry

from .helpers import MockLLM, call, calls, text

pytestmark = pytest.mark.django_db


def _turn(*replies, said="hello", conversation=None, stream=True):
    conversation = conversation or Conversation.objects.create()
    llm = MockLLM(*replies)
    reason = run_turn(conversation, said, llm=llm, tools=ToolRegistry(basic()), stream=stream)
    return conversation, llm, reason


def _kinds(conversation):
    return [e.kind for e in conversation.events.all()]


def test_an_answer_in_words_ends_the_turn():
    """What: the model replies in words; the person's message and the reply are
    kept, and the page is told the text and that the turn is over.
    How: one scripted text reply."""
    conversation, _, reason = _turn(text("Hi. What would you like to tune?"))

    assert reason == "answered"
    assert [m.role for m in conversation.messages.all()] == ["user", "assistant"]
    assert "".join(e.payload["text"] for e in conversation.events.filter(kind="text")) \
        == "Hi. What would you like to tune?"
    assert _kinds(conversation)[0] == "user" and _kinds(conversation)[-1] == "turn_end"


def test_a_tool_result_is_kept_and_sent_back_with_its_call():
    """What: a tool called, its answer kept in the history right after the call,
    and the model asked again with both — the history A3S dropped tool answers
    from, which strict servers refuse.
    How: the model profiles a demo dataset, then answers."""
    conversation, llm, reason = _turn(call("profile_dataset", demo="iris"),
                                      text("Iris has 150 rows."))

    assert reason == "answered"
    roles = [m.role for m in conversation.messages.all()]
    assert roles == ["user", "assistant", "tool", "assistant"]
    second = llm.calls[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == second[-1]["tool_call_id"]
    assert '"rows": 150' in second[-1]["content"]
    assert {"tool_call", "tool_result"} <= set(_kinds(conversation))


def test_a_question_with_choices_waits_for_the_person():
    """What: asking the person to choose ends the turn, waiting, and the page is
    given the question and its options.
    How: the model calls present_choices."""
    conversation, _, reason = _turn(call("present_choices", question="Which task?",
                                         options=["classification", "regression"]))

    assert reason == "waiting"
    choices = conversation.events.get(kind="choices").payload
    assert choices == {"question": "Which task?", "options": ["classification", "regression"]}


def test_every_call_in_a_reply_is_answered_even_after_a_question():
    """What: when a reply asks the person something and calls another tool
    besides, that call is answered as not done, so no call is left without an
    answer in the history.
    How: present_choices then explain_how in one reply."""
    conversation, _, reason = _turn(calls(
        ("present_choices", {"question": "Which?", "options": ["a", "b"]}),
        ("explain_how", {"action": "run"})))

    assert reason == "waiting"
    tools = [m for m in conversation.messages.all() if m.role == "tool"]
    assert len(tools) == 2 and "Not done" in tools[1].content


def test_a_refusal_is_told_to_the_model_and_recorded():
    """What: a tool the person may not use answers with why, the turn goes on,
    and the audit records the refusal.
    How: profile_dataset with no dataset in the conversation."""
    conversation, llm, reason = _turn(call("profile_dataset"), text("I need your dataset."))

    assert reason == "answered"
    assert "Refused" in llm.calls[1]["messages"][-1]["content"]
    assert AgentAction.objects.get(conversation=conversation).outcome == AgentAction.REFUSED


def test_arguments_a_tool_does_not_take_are_refused_before_it_runs():
    """What: arguments outside a tool's schema never reach it.
    How: explain_how with an action it does not know and an extra argument."""
    conversation, llm, _ = _turn(call("explain_how", action="format the disk"),
                                 call("explain_how", action="run", force=True),
                                 text("ok"))

    outcomes = list(AgentAction.objects.filter(conversation=conversation)
                    .values_list("outcome", flat=True))
    assert outcomes == [AgentAction.INVALID, AgentAction.INVALID]


def test_a_model_that_repeats_itself_is_stopped():
    """What: the same call three times in one turn ends it.
    How: the model asks how to run, three times."""
    conversation, _, reason = _turn(*[call("explain_how", action="run")] * 5)

    assert reason == "repeating"


def test_the_turn_stops_at_the_tool_call_limit(settings):
    """What: no more tool calls in one turn than ASSISTANT_MAX_TOOL_CALLS.
    How: a limit of two and a reply making three."""
    settings.ASSISTANT_MAX_TOOL_CALLS = 2
    conversation, _, reason = _turn(calls(("explain_how", {"action": "run"}),
                                          ("explain_how", {"action": "cancel"}),
                                          ("explain_how", {"action": "delete"})))

    assert reason == "tool_calls"
    assert AgentAction.objects.filter(conversation=conversation).count() == 2


def test_the_turn_stops_at_the_step_limit(settings):
    """What: no more model calls in one turn than ASSISTANT_MAX_STEPS.
    How: a limit of two and a model that keeps calling tools."""
    settings.ASSISTANT_MAX_STEPS = 2
    _, llm, reason = _turn(call("explain_how", action="run"), call("explain_how", action="cancel"),
                           call("explain_how", action="delete"))

    assert reason == "steps" and len(llm.calls) == 2


def test_a_spent_budget_ends_the_turn_before_the_model_is_asked(settings):
    """What: once a person's conversations have spent the day's tokens, the
    model is not asked again.
    How: a budget of 15 tokens, a first turn costing 10, then two more."""
    settings.ASSISTANT_DAILY_TOKENS = 15
    conversation, _, _ = _turn(text("one", tokens=10))
    _turn(text("two", tokens=10), conversation=conversation)
    _, llm, reason = _turn(text("three"), conversation=conversation)

    assert tokens_today(None) == 20
    assert reason == "budget" and llm.calls == []


def test_an_unreachable_model_ends_the_turn_quietly():
    """What: a model that cannot be reached ends the turn with a reason, not an
    error, and the person's message is kept.
    How: the scripted model raises `Unavailable`."""
    conversation, _, reason = _turn(Unavailable("connection refused"))

    assert reason == "unavailable"
    assert conversation.events.get(kind="turn_end").payload["detail"] == "connection refused"
    assert conversation.messages.get().role == "user"


def test_the_history_starts_with_the_persons_message():
    """What: the history sent to the model never starts with a tool answer or a
    call whose answer was cut off, and reaches back to the person's message
    even when one turn alone is longer than the window.
    How: a conversation of four turns, trimmed to a window of three."""
    import assistant.loop as loop

    conversation = Conversation.objects.create()
    for _ in range(4):
        _turn(call("explain_how", action="run"), text("done"), conversation=conversation)
    old = loop.HISTORY
    loop.HISTORY = 3
    try:
        trimmed = history(conversation)
    finally:
        loop.HISTORY = old

    assert [m["role"] for m in trimmed] == ["user", "assistant", "tool", "assistant"]
    assert Message.objects.filter(conversation=conversation).count() == 16
