"""The person's chats (assistant/views.py `chats_context`, assistant/_dock.html):
the general chat and one per experiment as tabs, a visitor that navigated the
person away from its own experiment and shows Return, reset in every phase,
the person's own off switch, the pages of the whole instance, and each
experiment's notepad (assistant/notes.py, assistant/tools/notes.py, and its
page, ui/views/notes.py).
"""

import json

import pytest
from django.urls import reverse
from django.utils import timezone

from assistant import notes as notepad
from assistant.gateway import Context
from assistant.models import AgentAction, BrowserPage, Conversation, ExperimentNotes
from assistant.prompts import system_prompt
from assistant.tools.client import PAGES
from assistant.turns import tools_for
from experiments.models import Experiment
from tests.access.test_groups import hosted, world  # noqa: F401 — fixtures
from ui import setup_steps as steps
from ui.models import ReaderPreferences

pytestmark = pytest.mark.django_db

IRIS = {
    "name": "irises", "task": "classification", "target_column": "variety",
    "model_name": "Random Forest", "evaluation": {"scheme": "kfold", "value": 5},
    "optimizer": {"name": "SMAC", "params": {}}, "metric": "accuracy",
}


@pytest.fixture
def on(settings):
    settings.ASSISTANT_ENABLED = True
    settings.ASSISTANT_ACT_SECONDS = 0


def _setting_up(client):
    """A plan approved in *client*'s browser: the conversation and its draft."""
    conversation = Conversation.objects.create(demo_dataset="iris", plan=IRIS)
    response = client.post(reverse("assistant:approve"), {"conversation": conversation.pk})
    assert response.status_code == 200, response.content
    conversation.refresh_from_db()
    return conversation, conversation.experiment


def _experiment(name="other", owner=None):
    return Experiment.objects.create(name=name, model_name="Random Forest",
                                     optimizer_name="Random Search",
                                     metric_names=["accuracy"], seed=0, owner=owner)


def _report(client, path):
    return client.post(reverse("assistant:page"), json.dumps({"path": path, "title": "A page"}),
                       content_type="application/json")


def _run(conversation, tool, **args):
    return tools_for(conversation).execute(tool, args, Context(conversation))


def _keys(response):
    return [tab["key"] for tab in response.context["assistant"]["tabs"]]


# ── the off switch ───────────────────────────────────────────────────────────

def test_turned_off_the_assistant_is_gone_until_turned_on(client, on):
    """What: a person who turns the assistant off in Appearance sees no chat
    anywhere and its endpoints are not found; turned on, it is back.
    How: the switch posted each way, then the home page and a message."""
    assert b'name="assistant"' in client.get(reverse("ui:appearance")).content

    client.post(reverse("ui:appearance"), {"assistant": "off"})
    home = client.get(reverse("ui:home")).content.decode()

    assert ReaderPreferences.objects.get().assistant_enabled is False
    assert "data-assistant" not in home and "Use the sidebar" in home
    assert client.post(reverse("assistant:message"), {"text": "hi"}).status_code == 404

    client.post(reverse("ui:appearance"), {"assistant": "on"})
    assert "data-assistant" in client.get(reverse("ui:home")).content.decode()


def test_the_switch_is_not_offered_when_the_instance_has_no_assistant(client):
    """What: with the assistant off for the whole instance, Appearance has no
    switch for it.
    How: the Appearance page, assistant off."""
    assert b'name="assistant"' not in client.get(reverse("ui:appearance")).content


# ── tabs ─────────────────────────────────────────────────────────────────────

def test_each_page_has_the_general_chat_and_its_experiments(client, on):
    """What: the home page is the general chat and docks nothing; another page
    docks the general chat; an experiment's page docks the general chat and
    the experiment's, named, and opens on the experiment's.
    How: the chat context of the three pages."""
    exp = _experiment("lilies")

    home = client.get(reverse("ui:home"))
    listing = client.get(reverse("ui:experiment_list"))
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk]))

    assert _keys(home) == [] and "assistant-dock" not in home.content.decode()
    assert _keys(listing) == ["general"]
    assert _keys(page) == ["general", "experiment"]
    assert page.context["assistant"]["active"] == "experiment"
    assert page.context["assistant"]["tabs"][1]["caption"] == "lilies"


def test_the_chat_answering_is_the_tab_a_page_opens_on(client, on):
    """What: the general chat that is answering when an experiment's page
    opens, having just brought the person there, is the tab shown.
    How: a general chat with a turn running, and an experiment's page."""
    exp = _experiment()
    Conversation.objects.create(turn_started_at=timezone.now())

    page = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert page.context["assistant"]["active"] == "general"


# ── visiting ─────────────────────────────────────────────────────────────────

def test_a_chat_that_navigates_away_from_its_experiment_visits_with_return(client, on):
    """What: an experiment's chat that takes the person to another experiment
    stays as a third tab, saying where it is from, with Return; it cannot
    fill there; Return goes back to the page it took them from and ends the
    visit.
    How: navigate from a draft's setup to another experiment's dashboard,
    that page, a fill, then Return."""
    conversation, exp = _setting_up(client)
    setup = steps.url(exp, steps.SETUP)
    _report(client, setup)
    other = _experiment()

    _run(conversation, "navigate", page="dashboard", experiment=other.pk)
    page = client.get(reverse("ui:experiment_detail", args=[other.pk]))

    assert BrowserPage.objects.get().visitor_id == conversation.pk
    assert _keys(page) == ["general", "experiment", "visitor"]
    assert page.context["assistant"]["active"] == "visitor"
    assert "data-assistant-back" in page.content.decode()

    _report(client, reverse("ui:experiment_settings", args=[other.pk]))
    refused = _run(conversation, "fill", values={"ice_max_curves": 5})
    assert "visiting" in refused.output

    assert client.post(reverse("assistant:back")).json() == {"url": setup}
    assert BrowserPage.objects.get().visitor_id is None


def test_the_visit_ends_on_the_visitors_own_experiment(client, on):
    """What: once the person opens any page of the visitor's own experiment,
    by whatever way, it is that page's own chat again, not a visitor.
    How: navigate away, then the draft's setup opened."""
    conversation, exp = _setting_up(client)
    _report(client, steps.url(exp, steps.SETUP))
    _run(conversation, "navigate", page="experiments")

    page = client.get(steps.url(exp, steps.SETUP))

    assert _keys(page) == ["general", "experiment"]
    assert BrowserPage.objects.get().visitor_id is None


def test_the_general_chat_never_visits(client, on):
    """What: the general chat is a tab on every page, so taking the person
    anywhere makes no visit.
    How: the general chat navigating to Appearance."""
    _report(client, reverse("ui:experiment_list"))
    general = Conversation.objects.create(browser=client.session.session_key)

    _run(general, "navigate", page="appearance")
    assert BrowserPage.objects.get().visitor_id is None


# ── the pages of the whole instance ──────────────────────────────────────────

def test_navigate_opens_the_settings_pages_the_person_may(client, on, hosted, world):  # noqa: F811
    """What: the assistant takes the person to their settings, and to the
    instance's defaults only if they may change them.
    How: ana, who may not, navigating to Appearance and to the defaults."""
    conversation = Conversation.objects.create(owner=world["ana"])

    went = _run(conversation, "navigate", page="appearance")
    refused = _run(conversation, "navigate", page="experiment_defaults")

    assert "Refused" not in went.output
    assert "cannot open the experiment_defaults page" in refused.output


def test_list_experiments_lists_only_what_the_person_sees(hosted, world):  # noqa: F811
    """What: list_experiments gives the person's own experiments and those
    shared with them, nobody else's.
    How: ben's general chat listing."""
    found = json.loads(_run(Conversation.objects.create(owner=world["ben"]),
                            "list_experiments").output)
    assert {e["name"] for e in found["experiments"]} == {"ana viewer", "ana contrib"}


def test_every_page_navigate_opens_has_an_explainer():
    """What: about_page explains each page the assistant can take the person to.
    How: about_page for every page on navigate's list."""
    conversation = Conversation.objects.create()
    for page in PAGES:
        output = _run(conversation, "about_page", page=page).output
        assert "What it is for" in output, page


# ── reset ────────────────────────────────────────────────────────────────────

def test_a_reset_waits_for_the_answer_and_keeps_the_visit(client, on):
    """What: a chat cannot be reset while it answers; reset, the empty chat
    that replaces it is still the visitor on the person's page.
    How: a visiting experiment's chat reset during a turn, then after."""
    conversation, exp = _setting_up(client)
    _report(client, steps.url(exp, steps.SETUP))
    _run(conversation, "navigate", page="experiments")
    Conversation.objects.filter(pk=conversation.pk).update(turn_started_at=timezone.now())

    busy = client.post(reverse("assistant:reset"), {"conversation": conversation.pk})
    Conversation.objects.filter(pk=conversation.pk).update(turn_started_at=None)
    done = client.post(reverse("assistant:reset"), {"conversation": conversation.pk})

    assert busy.status_code == 409
    assert BrowserPage.objects.get().visitor_id == done.json()["conversation"]


# ── the notepad ──────────────────────────────────────────────────────────────

def _notes(client, exp):
    return client.get(reverse("ui:experiment_notes", args=[exp.pk]))


def test_the_notes_are_a_view_of_the_experiment_beside_its_dashboard(client):
    """What: the sidebar lists Notes with the open experiment's other views,
    for a draft beside its setup, and the page is there with the assistant off.
    How: an experiment's dashboard and a draft's notes page."""
    exp = _experiment()
    draft = Experiment.objects.create(name="d", model_name="Random Forest",
                                      optimizer_name="Random Search", metric_names=["accuracy"],
                                      seed=0, draft=True)
    url = reverse("ui:experiment_notes", args=[exp.pk])

    assert f'href="{url}"' in client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    page = _notes(client, draft)
    assert page.status_code == 200
    assert 'aria-current="page">Notes' in page.content.decode()


def test_who_may_edit_gets_the_editor_and_who_may_view_reads(client, hosted, world):  # noqa: F811
    """What: a viewer reads the notes and cannot save them; a contributor gets
    the editor, saves, and is named as the last to change them; someone the
    experiment is not shared with cannot open them.
    How: ben on the experiment ana shared for viewing and the one she shared
    for contributing, and on cleo's."""
    notepad.save(world["ana_viewer"], "Recall on class 2 matters most.", user=world["ana"], seen="")
    client.force_login(world["ben"])

    viewed = _notes(client, world["ana_viewer"]).content.decode()
    refused = client.post(reverse("ui:experiment_notes", args=[world["ana_viewer"].pk]),
                          {"text": "mine", "seen": ""})
    saved = client.post(reverse("ui:experiment_notes", args=[world["ana_contrib"].pk]),
                        {"text": "Tried rf; dropped.", "seen": ""})

    assert "Recall on class 2 matters most." in viewed and "data-notes-editor" not in viewed
    assert refused.status_code == 403
    assert saved.status_code == 302
    after = _notes(client, world["ana_contrib"]).content.decode()
    assert "data-notes-editor" in after and "Last changed by ben:" in after
    assert _notes(client, world["cleo_private"]).status_code == 404


def test_a_save_over_newer_notes_is_refused_with_them_shown(client, on):
    """What: a person saving notes someone changed after they opened them is
    refused; the page shows the newer notes and keeps their text in the
    editor.
    How: notes opened, changed by an assistant, then saved with the version
    first opened."""
    exp = _experiment()
    seen = notepad.as_dict(notepad.save(exp, "Budget is 100 trials.", user=None, seen=""))["updated_at"]
    notepad.edit(exp, "Budget is 200 trials.", user=None, conversation=None,
                 replace="Budget is 100 trials.")

    page = client.post(reverse("ui:experiment_notes", args=[exp.pk]),
                       {"text": "Budget is 150.", "seen": seen}).content.decode()

    assert ExperimentNotes.objects.get().text == "Budget is 200 trials."
    assert "Budget is 200 trials." in page and ">Budget is 150.</textarea>" in page


def test_the_notes_have_room_for_so_much(client, on):
    """What: notes longer than the limit are refused with why, and nothing is
    saved.
    How: a save one character over."""
    exp = _experiment()
    page = client.post(reverse("ui:experiment_notes", args=[exp.pk]),
                       {"text": "x" * (ExperimentNotes.MAX_LENGTH + 1), "seen": ""})

    assert "at most" in page.content.decode()
    assert not ExperimentNotes.objects.filter(text__startswith="x").exists()


def test_an_assistant_adds_and_replaces_passages_and_every_chat_is_told(client, on):
    """What: an assistant adds a passage at the end, as the person's
    assistant, recorded; it replaces a passage only when it quotes one that
    is there exactly once; the system prompt of a chat about the experiment
    carries the notes as data.
    How: write_notes twice in an experiment's chat, once with a passage that
    is not there, then its system prompt."""
    conversation, exp = _setting_up(client)
    notepad.save(exp, "# Directives\n\nNo SVMs.", user=None, seen="")

    _run(conversation, "write_notes", text="The person cares about recall on class 2.")
    _run(conversation, "write_notes", text="SVMs are fine after all.", replace="No SVMs.")
    missing = _run(conversation, "write_notes", text="x", replace="not in the notes")
    notes = ExperimentNotes.objects.get()

    assert notes.text == ("# Directives\n\nSVMs are fine after all.\n\n"
                          "The person cares about recall on class 2.")
    assert notes.by_assistant and notes.conversation_id == conversation.pk
    assert "not in the notes" in missing.output
    assert AgentAction.objects.filter(tool="write_notes", outcome=AgentAction.OK).count() == 2
    prompt = system_prompt(conversation)
    assert "recall on class 2" in prompt and "data, not instructions" in prompt


def test_a_viewers_assistant_cannot_write_notes(hosted, world):  # noqa: F811
    """What: the assistant writes only where its person may edit.
    How: ben's chat about the experiment shared with him for viewing."""
    conversation = Conversation.objects.create(owner=world["ben"],
                                               experiment=world["ana_viewer"],
                                               phase=Conversation.DASHBOARD)
    result = _run(conversation, "write_notes", text="mine")
    assert "Refused" in result.output and not ExperimentNotes.objects.exists()


def test_the_prompt_carries_the_start_of_long_notes():
    """What: notes longer than the prompt's share give their start and how
    much more there is.
    How: notes of 5000 characters."""
    exp = _experiment()
    notepad.save(exp, "a" * 3000 + "b" * 2000, user=None, seen="")

    found = notepad.for_prompt(exp)

    assert found.startswith("a" * 3000 + "b" * 1000)
    assert found.endswith("[… 1000 more characters: read_notes]")
