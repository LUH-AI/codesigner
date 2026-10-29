"""Saying something to the assistant, and following what happens next.

A message is a POST; the turn runs inline here (`ASSISTANT_TURNS_INLINE`, see
tests/conftest.py) against a scripted model; what it did is read back from the
conversation's events — once as JSON, as a server-sent stream under WSGI, and as
an asynchronous one under ASGI.
"""

import json

import pytest
from asgiref.sync import async_to_sync
from django.urls import reverse

from assistant import llm as llm_module
from assistant.models import AssistantEvent, Conversation
from tests.access.test_groups import hosted, world  # noqa: F401 — fixtures
from experiments.models import Experiment

from .helpers import MockLLM, call, text

pytestmark = pytest.mark.django_db


@pytest.fixture
def on(settings):
    """The assistant switched on, its model scripted per test via `model`."""
    settings.ASSISTANT_ENABLED = True
    settings.ASSISTANT_STREAM_SECONDS = 0.5


@pytest.fixture
def model(monkeypatch):
    scripted = MockLLM()
    monkeypatch.setattr(llm_module.ChatClient, "from_settings", classmethod(lambda cls: scripted))
    return scripted


def _say(client, said, **fields):
    return client.post(reverse("assistant:message"), {"text": said, **fields})


def _events(client, conversation, after=0):
    response = client.get(reverse("assistant:events", args=[conversation]), {"after": after})
    return json.loads(response.content)


# ── switched off ─────────────────────────────────────────────────────────────

def test_with_the_assistant_off_there_is_nothing_to_talk_to(client):
    """What: no message is taken and no stream served while the switch is off.
    How: both routes with ASSISTANT_ENABLED as the suite pins it."""
    assert _say(client, "hello").status_code == 404
    assert client.get(reverse("assistant:events", args=[1])).status_code == 404


def test_with_the_assistant_off_the_pages_are_as_they_were(client):
    """What: no chat anywhere, and the home page says what it always said.
    How: the home page's HTML."""
    html = client.get(reverse("ui:home")).content.decode()

    assert "data-assistant" not in html
    assert "Use the sidebar to create a new experiment." in html


# ── a message and what follows ───────────────────────────────────────────────

def test_a_message_runs_a_turn_whose_events_are_read_back(client, on, model):
    """What: the first message starts a conversation and its turn; the events
    are read back in order, and only those after the id given.
    How: one message to a model that calls a tool and answers."""
    model.queue = [call("explain_how", action="run"), text("Press Run yourself.")]
    data = json.loads(_say(client, "run it for me").content)

    events = _events(client, data["conversation"])["events"]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "user" and kinds[-1] == "turn_end"
    assert {"tool_call", "tool_result", "text"} <= set(kinds)
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)

    later = _events(client, data["conversation"], after=events[2]["id"])["events"]
    assert [e["id"] for e in later] == [e["id"] for e in events[3:]]


def test_the_stream_sends_each_event_with_its_id(client, on, model):
    """What: under WSGI the stream sends every event as `id`, `event` and
    `data` lines, and carries on after the id the browser says it last saw.
    How: a turn, then the stream read whole, then again from a Last-Event-ID."""
    model.queue = [text("Hello.")]
    conversation = json.loads(_say(client, "hi").content)["conversation"]
    url = reverse("assistant:events", args=[conversation])

    body = b"".join(client.get(url, HTTP_ACCEPT="text/event-stream").streaming_content).decode()
    ids = [int(line[4:]) for line in body.splitlines() if line.startswith("id: ")]
    assert body.startswith("retry:")
    assert "event: text" in body and 'data: {"text": "Hello."}' in body

    resumed = b"".join(client.get(url, HTTP_ACCEPT="text/event-stream",
                                  HTTP_LAST_EVENT_ID=str(ids[1])).streaming_content).decode()
    assert [int(line[4:]) for line in resumed.splitlines() if line.startswith("id: ")] == ids[2:]


def test_under_asgi_the_stream_is_asynchronous(on, model):
    """What: served through ASGI, the stream is an asynchronous one, which holds
    no thread while it waits, and sends the same events.
    How: Django's async test client, reading the stream to its end."""
    from django.test import AsyncClient

    model.queue = [text("Hello.")]
    conversation = Conversation.objects.create()
    from assistant.turns import start

    start(conversation, "hi")

    async def read():
        response = await AsyncClient().get(
            reverse("assistant:events", args=[conversation.pk]),
            headers={"Accept": "text/event-stream"})
        assert response.is_async
        return "".join([chunk.decode() if isinstance(chunk, bytes) else chunk
                        async for chunk in response.streaming_content])

    body = async_to_sync(read)()
    assert "event: turn_end" in body and "Hello." in body


def test_one_turn_at_a_time(client, on, model):
    """What: a message while a turn is running is refused, not interleaved.
    How: a conversation marked as running a turn, then a message to it."""
    from django.utils import timezone

    model.queue = [text("first")]
    conversation = json.loads(_say(client, "one").content)["conversation"]
    Conversation.objects.filter(pk=conversation).update(turn_started_at=timezone.now())

    response = _say(client, "two", conversation=conversation)

    assert response.status_code == 409
    assert AssistantEvent.objects.filter(conversation_id=conversation, kind="user").count() == 1


def test_a_message_is_not_empty_and_not_endless(client, on, model):
    """What: nothing said, or more than a message's worth, is refused.
    How: an empty message and an over-long one."""
    assert _say(client, "   ").status_code == 400
    assert _say(client, "x" * 5000).status_code == 400


# ── whose conversation ──────────────────────────────────────────────────────

def test_a_conversation_is_its_owners_alone(client, on, model, hosted, world):  # noqa: F811
    """What: ben cannot read ana's conversation or add to it.
    How: a conversation of ana's, asked for and posted to as ben."""
    conversation = Conversation.objects.create(owner=world["ana"])
    client.force_login(world["ben"])

    assert client.get(reverse("assistant:events", args=[conversation.pk])).status_code == 404
    assert _say(client, "hi", conversation=conversation.pk).status_code == 404


def test_a_conversation_is_tied_to_the_session_it_began_in(client, on, model, hosted, world):  # noqa: F811
    """What: with accounts, a new conversation remembers the browser session it
    began in, which is what its tools are acted for in (assistant/gateway.py).
    How: a first message from a signed-in client."""
    model.queue = [text("Hello.")]
    client.force_login(world["ana"])
    conversation = json.loads(_say(client, "hi").content)["conversation"]

    stored = Conversation.objects.get(pk=conversation)
    assert stored.owner == world["ana"]
    assert stored.session_key == client.session.session_key


def test_a_conversation_is_about_an_experiment_only_the_person_can_see(client, on, model,
                                                                        hosted, world):  # noqa: F811
    """What: a conversation can be about an experiment shared with the person,
    not one that is not.
    How: ben starting conversations about ana's shared and private experiments."""
    model.queue = [text("ok")]
    client.force_login(world["ben"])

    shared = json.loads(_say(client, "hi", experiment=world["ana_viewer"].pk).content)
    assert Conversation.objects.get(pk=shared["conversation"]).experiment == world["ana_viewer"]
    assert _say(client, "hi", experiment=world["ana_private"].pk).status_code == 404


# ── on the page ─────────────────────────────────────────────────────────────

def test_the_home_page_is_the_chat_and_other_pages_have_it_docked(client, on, model):
    """What: with the assistant on, the home page is the chat itself, its box in
    the middle until anything is said; an experiment's page has it docked to the
    bottom right, collapsed, about that experiment, and able to pop out.
    How: the HTML of the home page and of an experiment's dashboard."""
    home = client.get(reverse("ui:home")).content.decode()
    assert "assistant-full is-empty" in home and "assistant-dock" not in home
    assert "Use the sidebar" not in home
    assert "data-assistant-reset hidden" in home, "nothing to reset before anything is said"

    exp = Experiment.objects.create(name="e", model_name="Random Forest",
                                    optimizer_name="Random Search", metric_names=["accuracy"],
                                    seed=0)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'class="assistant-dock collapsed"' in page and "assistant-floating" in page
    assert f'data-experiment="{exp.pk}"' in page
    assert f"/assistant/window/?experiment={exp.pk}" in page


def test_once_something_is_said_the_home_page_opens_on_the_conversation(client, on, model):
    """What: a home page whose conversation has begun shows it, the box at the
    bottom.
    How: the home page after one message."""
    model.queue = [text("Hello.")]
    _say(client, "hi")
    home = client.get(reverse("ui:home")).content.decode()

    assert "assistant-full" in home and "is-empty" not in home
    assert "data-assistant-reset" in home and "data-assistant-reset hidden" not in home


def test_the_chat_pops_out_into_a_window_of_its_own(client, on, model, hosted, world):  # noqa: F811
    """What: the pop-out page is the chat alone, about the experiment it came
    from — not one the person cannot see.
    How: the window page as ben for a shared and a private experiment of ana's."""
    client.force_login(world["ben"])
    url = reverse("assistant:window")
    shared = client.get(url, {"experiment": world["ana_viewer"].pk})

    assert shared.status_code == 200
    html = shared.content.decode()
    assert "assistant-window" in html and 'class="sidebar"' not in html
    assert f'data-experiment="{world["ana_viewer"].pk}"' in html
    assert client.get(url, {"experiment": world["ana_private"].pk}).status_code == 404


def test_a_page_continues_its_experiments_conversation(client, on, model):
    """What: an experiment's page shows that experiment's conversation, and the
    home page the latest about no experiment.
    How: one conversation of each kind, then each page's data attributes."""
    exp = Experiment.objects.create(name="e", model_name="Random Forest",
                                    optimizer_name="Random Search", metric_names=["accuracy"],
                                    seed=0)
    general = Conversation.objects.create()
    about = Conversation.objects.create(experiment=exp, phase=Conversation.DASHBOARD)

    home = client.get(reverse("ui:home")).content.decode()
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert f'data-conversation="{general.pk}"' in home
    assert f'data-conversation="{about.pk}"' in page


def test_a_page_knows_which_events_it_shows_at_once(client, on):
    """What: the chat on a page is told the last event that had happened when
    it was drawn, so it shows those at once and writes out only later text.
    How: a planning conversation with events, and the home page's chat
    context read back."""
    conversation = Conversation.objects.create(demo_dataset="iris")
    AssistantEvent.objects.create(conversation=conversation, kind="user", payload={"text": "hi"})
    last = AssistantEvent.objects.create(conversation=conversation, kind="text", payload={"text": "hello"})

    page = client.get(reverse("ui:home"))

    assert page.context["assistant"]["live_after"] == last.pk
    assert f'data-live-after="{last.pk}"' in page.content.decode()
