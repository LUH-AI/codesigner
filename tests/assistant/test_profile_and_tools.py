"""What the assistant is told about a dataset, and the tools every conversation
has: worked out here from the data, and only as much of the data as the
instance allows."""

import json

import pytest
from django.core.management import call_command

from assistant.gateway import Context
from assistant.models import Conversation
from assistant.profile import profile
from assistant.tools.basic import HOW, basic
from assistant.tools.registry import ToolRegistry, invalid
from tests.conftest import DATASETS_DIR

from .helpers import MockLLM, call, text

pytestmark = pytest.mark.django_db


def test_a_dataset_is_described_by_its_facts():
    """What: iris's rows, its columns' kinds, its target and the task that
    suggests, and how its classes are balanced.
    How: the profile of the bundled iris."""
    facts = profile(DATASETS_DIR / "iris.csv")

    assert facts["rows"] == 150
    assert [c["kind"] for c in facts["columns"]] == ["numeric"] * 4
    assert facts["target"]["name"] == "variety"
    assert facts["target"]["suggested_task"] == "classification"
    assert facts["target"]["smallest_class_share"] == pytest.approx(1 / 3, abs=0.01)


def test_a_chosen_target_is_described_as_the_target():
    """What: with another column chosen, that one is the target and the old last
    column a feature.
    How: iris profiled predicting petal width."""
    facts = profile(DATASETS_DIR / "iris.csv", "petal.width")

    assert facts["target"]["name"] == "petal.width" and facts["target"]["chosen"]
    assert facts["target"]["suggested_task"] == "regression"
    assert "variety" in [c["name"] for c in facts["columns"]]


def test_no_values_leave_the_machine_when_none_may(settings):
    """What: with ASSISTANT_SAMPLE_ROWS at 0 the model gets names, kinds and
    counts, and not a single value of the data.
    How: iris profiled both ways, looking for its values."""
    settings.ASSISTANT_SAMPLE_ROWS = 3
    shown = json.dumps(profile(DATASETS_DIR / "iris.csv"))
    settings.ASSISTANT_SAMPLE_ROWS = 0
    hidden = json.dumps(profile(DATASETS_DIR / "iris.csv"))

    assert "Setosa" in shown and "examples" in shown
    assert "Setosa" not in hidden and "examples" not in hidden


def test_columns_a_series_could_be_ordered_by_are_named():
    """What: the airline series names its month as a column to order by.
    How: its profile."""
    assert "month" in profile(DATASETS_DIR / "airline.csv")["time_order_candidates"]


def test_every_human_only_action_has_its_explanation():
    """What: explain_how covers everything only the person may do, each in the
    pages' own words.
    How: its table against the actions the plan names."""
    assert {"run", "cancel", "delete", "discard", "share", "transfer", "groups",
            "password", "account", "custom_model", "defaults"} == set(HOW)


def test_the_tools_offered_are_described_in_a_form_the_registry_checks():
    """What: every tool's schema is one `invalid` understands, and refuses an
    argument the tool does not take.
    How: each basic tool's schema with an extra argument."""
    for spec in basic():
        assert invalid({"__extra__": 1}, spec.parameters)
    assert invalid({"action": 3}, basic()[2].parameters) == "action must be a string"
    assert invalid({"question": "q", "options": ["a", 2]}, basic()[1].parameters)


def test_a_dataset_is_reached_through_the_conversations_experiment():
    """What: profile_dataset without a demo reads the conversation's own
    experiment's data, through the gateway.
    How: a conversation about a draft with iris staged on it."""
    from django.contrib.auth.models import AnonymousUser
    from django.test import RequestFactory

    from ui import drafts

    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    exp = drafts.new(request)
    drafts.stage_dataset(request, exp, demo="iris")
    conversation = Conversation.objects.create(experiment=exp)

    result = ToolRegistry(basic()).execute("profile_dataset", {}, Context(conversation))

    assert json.loads(result.output)["rows"] == 150
    assert result.experiment_id == exp.pk


def test_the_terminal_chat_talks_to_the_model(monkeypatch, capsys):
    """What: `manage.py assistant_chat` runs a turn per line typed and prints
    what the assistant did and said.
    How: the command with a scripted model and two lines of input."""
    import builtins

    from assistant import llm as llm_module

    model = MockLLM(call("explain_how", action="run"), text("Press Run yourself."))
    monkeypatch.setattr(llm_module.ChatClient, "from_settings", classmethod(lambda cls: model))
    lines = iter(["can you run it?", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))

    call_command("assistant_chat")
    out = capsys.readouterr().out

    assert "explain_how" in out and "Press Run yourself." in out
