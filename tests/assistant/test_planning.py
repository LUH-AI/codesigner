"""Planning an experiment in the chat: the dataset chosen first, the plan
written down as it is decided and checked against the setup's own forms, and
the person approving it — after which the setup is what comes next.

Scripted models stand in for a real one (helpers.MockLLM).
"""

import json

import pytest
from django.contrib.auth.models import AnonymousUser
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from django.urls import reverse

from assistant import llm as llm_module
from assistant import plan as plans
from assistant.gateway import Context
from assistant.models import AgentAction, Conversation
from assistant.turns import tools_for
from tests.access.test_groups import hosted, world  # noqa: F401 — fixtures
from ui import setup_steps as steps

from .helpers import MockLLM, call, calls, text

pytestmark = pytest.mark.django_db

#: A plan for iris that the setup accepts as it stands.
IRIS = {
    "goal": "Tell the three irises apart.",
    "name": "irises",
    "task": "classification",
    "target_column": "variety",
    "model_name": "Random Forest",
    "evaluation": {"scheme": "kfold", "value": 5},
    "optimizer": {"name": "SMAC", "params": {}},
    "metric": "accuracy",
    "stopping": {"max_trials": 30},
}


@pytest.fixture
def on(settings):
    settings.ASSISTANT_ENABLED = True


@pytest.fixture
def model(monkeypatch):
    scripted = MockLLM()
    monkeypatch.setattr(llm_module.ChatClient, "from_settings", classmethod(lambda cls: scripted))
    return scripted


def _request():
    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    return request


def _planning(demo="iris", **fields):
    """A conversation planning for a demo dataset; no draft until approval."""
    return Conversation.objects.create(demo_dataset=demo, **fields)


def _iris():
    from core.datasets import demo_datasets
    from pathlib import Path

    return Path(demo_datasets()["iris"])


# ── the plan, checked as the setup would ─────────────────────────────────────

def test_a_change_to_the_plan_is_laid_over_it_field_by_field():
    """What: a section changed in part keeps the rest of what it said.
    How: `merged` on an evaluation and an optimizer's settings."""
    plan = {"evaluation": {"scheme": "kfold", "value": 5},
            "optimizer": {"name": "SMAC", "params": {"search_strategy": "gp"}}}
    changed = plans.merged(plan, {"evaluation": {"value": 10},
                                  "optimizer": {"params": {"trial_cap": 20}}})

    assert changed["evaluation"] == {"scheme": "kfold", "value": 10}
    assert changed["optimizer"]["params"] == {"search_strategy": "gp", "trial_cap": 20}


def test_a_plan_the_setup_accepts_has_no_problems():
    """What: iris as a classification with a forest, SMAC and 30 trials can be
    set up as it stands.
    How: `problems` on the plan against a draft with iris on it."""
    assert plans.problems(IRIS, _request(), _iris()) == []


@pytest.mark.parametrize("change, expected", [
    ({"task": "regression"}, "task"),
    ({"model_name": "Logistic Regression", "task": "regression", "target_column": "petal.width"},
     "model_name"),
    ({"evaluation": {"scheme": "backtest", "value": 3}}, "time_column"),
    ({"target_column": "petal.colour"}, "target_column"),
    ({"optimizer": {"name": "SMAC", "params": {"warp_speed": 9}}}, "warp_speed"),
    ({"optimizer": {"name": "Annealing"}}, "no optimizer"),
    ({"model_name": "SVM", "processing": {"missing": "keep"}}, "cannot take"),
    ({"metric": "rmse"}, "rmse"),
    ({"stopping": {"max_trials": 0}}, "would end a run"),
])
def test_a_plan_the_setup_would_refuse_says_why(change, expected):
    """What: each kind of mistake the setup would catch is reported in its words
    before anything is saved: a task the target cannot be, a model that does not
    do the task, a forecast without a time column, a column that is not there, a
    setting or optimizer that does not exist, data handling the model cannot
    take, a metric of the other task, and a budget that ends nothing.
    How: `problems` on the iris plan with one thing changed."""
    found = plans.problems(plans.merged(IRIS, change), _request(), _iris())

    assert any(expected in problem for problem in found), found


def test_nothing_is_made_by_checking():
    """What: checking a plan makes no experiment, draft or otherwise.
    How: the experiments before and after `problems`."""
    from experiments.models import Experiment

    plans.problems(IRIS, _request(), _iris())

    assert not Experiment.objects.exists()


# ── the tools ────────────────────────────────────────────────────────────────

def test_planning_offers_what_codesigner_can_do():
    """What: the models for a task, the metrics it is scored by, and the
    optimizers with their settings, as the registries have them.
    How: the three listing tools through the registry."""
    tools = tools_for(Conversation.objects.create())
    context = Context(Conversation.objects.create())
    models = json.loads(tools.execute("list_models", {"task": "classification"}, context).output)
    metrics = tools.execute("list_metrics", {"task": "regression"}, context).output
    optimizers = json.loads(tools.execute("list_optimizers", {}, context).output)

    names = [m["name"] for m in models]
    assert "Random Forest" in names and "Elastic Net" not in names
    assert '"rmse"' in metrics
    smac = next(o for o in optimizers if o["name"] == "SMAC")
    assert "search_strategy" in [s["name"] for s in smac["settings"]]


def test_the_plan_is_written_down_and_shown_with_what_stops_it():
    """What: update_plan merges what was decided into the conversation's plan
    and shows the person the plan with what still stops it being set up.
    How: two updates through the registry on a conversation about a draft."""
    conversation = _planning()
    tools, context = tools_for(conversation), Context(conversation)

    first = tools.execute("update_plan", {"task": "classification"}, context)
    assert "Not ready" in first.output
    tools.execute("update_plan", {k: v for k, v in IRIS.items() if k != "task"}, context)
    conversation.refresh_from_db()

    assert conversation.plan["model_name"] == "Random Forest"
    shown = conversation.events.filter(kind="plan").last().payload
    assert shown["problems"] == [] and shown["plan"]["task"] == "classification"
    assert "can be set up" in tools.execute("check_plan", {}, context).output


def test_the_plan_takes_only_its_own_fields():
    """What: a field the plan does not have is refused before it is written.
    How: update_plan with a made-up field."""
    conversation = _planning()
    tools_for(conversation).execute("update_plan", {"rocket": True}, Context(conversation))

    assert AgentAction.objects.get(conversation=conversation).outcome == AgentAction.INVALID
    conversation.refresh_from_db()
    assert conversation.plan == {}


def test_without_a_dataset_there_is_no_plan_to_write():
    """What: before a dataset is chosen, update_plan says to ask for one.
    How: update_plan in a conversation about nothing."""
    conversation = Conversation.objects.create()
    result = tools_for(conversation).execute("update_plan", {"task": "regression"},
                                             Context(conversation))

    assert "dataset" in result.output


def test_once_approved_the_planning_tools_are_gone():
    """What: a conversation past planning is not offered the planning tools.
    How: `tools_for` in each phase."""
    assert "update_plan" in tools_for(Conversation(phase=Conversation.PLANNING))
    assert "update_plan" not in tools_for(Conversation(phase=Conversation.SETUP))


# ── choosing the dataset ─────────────────────────────────────────────────────

def test_choosing_a_dataset_makes_no_draft(client, on):
    """What: the dataset chosen is kept with a new conversation, and the chat is
    told its name; no draft exists until the plan is approved.
    How: a demo chosen on the home page, then the experiments."""
    from experiments.models import Experiment

    data = json.loads(client.post(reverse("assistant:dataset"), {"demo": "wine"}).content)
    conversation = Conversation.objects.get(pk=data["conversation"])

    assert data["dataset"] == "wine" and conversation.demo_dataset == "wine"
    assert conversation.experiment is None and not Experiment.objects.exists()
    assert conversation.events.get(kind="dataset").payload == {"name": "wine", "demo": True}


def test_an_uploaded_dataset_is_checked_as_step_one_checks_it(client, on):
    """What: a CSV is taken; a file that is not one is refused with why.
    How: an upload of each through the dataset box."""
    good = SimpleUploadedFile("rows.csv", b"a,b,y\n1,2,0\n3,4,1\n", content_type="text/csv")
    bad = SimpleUploadedFile("rows.txt", b"nope", content_type="text/plain")
    taken = json.loads(client.post(reverse("assistant:dataset"), {"dataset_file": good}).content)
    refused = client.post(reverse("assistant:dataset"),
                          {"dataset_file": bad, "conversation": taken["conversation"]})

    assert taken["dataset"] == "rows.csv"
    assert refused.status_code == 400 and ".csv" in json.loads(refused.content)["error"]


def test_the_home_page_shows_the_dataset_box_while_planning(client, on):
    """What: the home page's chat offers the demos, and once one is chosen shows
    it selected, with the demo box ticked.
    How: the home page before and after choosing iris."""
    before = client.get(reverse("ui:home")).content.decode()
    assert 'id="assistant-demo"' in before and '<option value="iris">' in before

    client.post(reverse("assistant:dataset"), {"demo": "iris"})
    after = client.get(reverse("ui:home")).content.decode()
    assert '<option value="iris" selected>' in after
    assert 'id="assistant-use-demo" checked' in after and "data-has-dataset" in after


def test_a_dataset_cannot_be_put_in_somebody_elses_conversation(client, on, hosted, world):  # noqa: F811
    """What: ben cannot choose a dataset for ana's conversation.
    How: a post naming ana's conversation, as ben."""
    conversation = Conversation.objects.create(owner=world["ana"])
    client.force_login(world["ben"])

    response = client.post(reverse("assistant:dataset"),
                           {"demo": "iris", "conversation": conversation.pk})
    assert response.status_code == 404


# ── approving ────────────────────────────────────────────────────────────────

def test_a_plan_with_problems_cannot_be_approved(client, on):
    """What: the approve button refuses a plan the setup would not accept, and
    says why.
    How: approving a conversation whose plan has no model."""
    conversation = _planning(plan={"task": "classification"})
    response = client.post(reverse("assistant:approve"), {"conversation": conversation.pk})

    assert response.status_code == 400
    assert json.loads(response.content)["problems"]
    conversation.refresh_from_db()
    assert conversation.phase == Conversation.PLANNING


def test_approving_makes_the_draft_and_moves_on_to_the_setup(client, on):
    """What: only an approved plan makes the draft, with the dataset on it;
    the setup's first step is where the person goes next, and the draft's pages
    continue the same conversation while the home page starts afresh.
    How: approving the iris plan, then the home page and step 1."""
    conversation = _planning(plan=IRIS)
    data = json.loads(client.post(reverse("assistant:approve"),
                                  {"conversation": conversation.pk}).content)
    conversation.refresh_from_db()
    exp = conversation.experiment

    assert exp.draft and exp.data.demo_dataset == "iris"
    assert conversation.demo_dataset == ""
    assert data["url"] == steps.url(exp, steps.SETUP)
    assert conversation.phase == Conversation.SETUP
    assert conversation.events.filter(kind="approved").exists()
    home = client.get(reverse("ui:home")).content.decode()
    setup = client.get(data["url"]).content.decode()
    assert 'data-conversation=""' in home
    assert f'data-conversation="{conversation.pk}"' in setup


def test_a_plan_is_approved_once(client, on):
    """What: an approved conversation has no plan left to approve, and takes no
    other dataset.
    How: approving twice, then choosing a dataset."""
    conversation = _planning(plan=IRIS)
    client.post(reverse("assistant:approve"), {"conversation": conversation.pk})

    again = client.post(reverse("assistant:approve"), {"conversation": conversation.pk})
    other = client.post(reverse("assistant:dataset"),
                        {"demo": "wine", "conversation": conversation.pk})
    assert again.status_code == 400 and other.status_code == 400


# ── a whole planning conversation ───────────────────────────────────────────

def test_a_conversation_plans_iris_and_is_approved(client, on, model):
    """What: the person chooses iris and says what they want; the assistant
    reads the data, asks which column to predict, writes the plan down, checks
    it and says it is ready; the person approves it and is sent to step 1.
    How: the endpoints a page uses, with a scripted model."""
    started = json.loads(client.post(reverse("assistant:dataset"), {"demo": "iris"}).content)
    conversation = started["conversation"]

    model.queue = [call("profile_dataset"),
                   call("present_choices", question="Which column should be predicted?",
                        options=["variety", "petal.width"])]
    client.post(reverse("assistant:message"),
                {"text": "I want to tell iris species apart", "conversation": conversation})
    model.queue = [calls(("update_plan", IRIS)), call("check_plan"),
                   text("The plan is ready: approve it to set it up.")]
    client.post(reverse("assistant:message"), {"text": "variety", "conversation": conversation})

    events = json.loads(client.get(reverse("assistant:events", args=[conversation]),
                                   {"after": 0}).content)["events"]
    kinds = [e["kind"] for e in events]
    assert kinds.count("turn_end") == 2
    assert "choices" in kinds and "plan" in kinds
    assert '"rows": 150' in model.calls[1]["messages"][-1]["content"]

    approved = json.loads(client.post(reverse("assistant:approve"),
                                      {"conversation": conversation}).content)
    assert approved["url"].endswith("/setup/")


# ── starting again ──────────────────────────────────────────────────────────

def test_reset_deletes_the_chat_and_its_upload_but_not_the_audit(client, on):
    """What: resetting a planning conversation deletes it and the file it was
    planning with, while the record of what the assistant did stays.
    How: a CSV uploaded, a tool call recorded, then a reset."""
    from pathlib import Path

    upload = SimpleUploadedFile("rows.csv", b"a,b,y\n1,2,0\n3,4,1\n", content_type="text/csv")
    started = json.loads(client.post(reverse("assistant:dataset"), {"dataset_file": upload}).content)
    conversation = Conversation.objects.get(pk=started["conversation"])
    stored = Path(conversation.dataset.path)
    tools_for(conversation).execute("check_plan", {}, Context(conversation))

    assert client.post(reverse("assistant:reset"),
                       {"conversation": conversation.pk}).status_code == 200
    assert not Conversation.objects.filter(pk=conversation.pk).exists()
    assert not stored.exists()
    assert AgentAction.objects.filter(tool="check_plan", conversation=None).count() == 1


def test_an_uploaded_dataset_goes_onto_the_draft_at_approval(client, on):
    """What: a CSV uploaded while planning becomes the draft's dataset when the
    plan is approved, and the conversation's own copy goes.
    How: an upload, a plan for it, then approval."""
    from pathlib import Path

    upload = SimpleUploadedFile("rows.csv", b"a,b,y\n" + b"1,2,0\n3,4,1\n" * 10,
                                content_type="text/csv")
    started = json.loads(client.post(reverse("assistant:dataset"), {"dataset_file": upload}).content)
    conversation = Conversation.objects.get(pk=started["conversation"])
    held = Path(conversation.dataset.path)
    conversation.plan = {**IRIS, "target_column": "y", "evaluation": {"scheme": "holdout", "value": 0.2}}
    conversation.save()

    response = client.post(reverse("assistant:approve"), {"conversation": conversation.pk})
    conversation.refresh_from_db()

    assert response.status_code == 200, response.content
    assert conversation.experiment.data.dataset.name.endswith(".csv")
    assert not held.exists() and not conversation.dataset


def test_an_approved_conversation_is_not_reset(client, on):
    """What: once the plan is approved, its draft is the person's to keep or
    discard on its own pages; reset refuses.
    How: approving the iris plan, then resetting."""
    conversation = _planning(plan=IRIS)
    client.post(reverse("assistant:approve"), {"conversation": conversation.pk})

    response = client.post(reverse("assistant:reset"), {"conversation": conversation.pk})
    assert response.status_code == 400
    assert Conversation.objects.filter(pk=conversation.pk).exists()


def test_somebody_elses_conversation_is_not_reset(client, on, hosted, world):  # noqa: F811
    """What: ben cannot reset ana's conversation.
    How: a reset naming it, as ben."""
    conversation = Conversation.objects.create(owner=world["ana"])
    client.force_login(world["ben"])

    assert client.post(reverse("assistant:reset"),
                       {"conversation": conversation.pk}).status_code == 404
    assert Conversation.objects.filter(pk=conversation.pk).exists()


def test_an_upload_keeps_its_own_name(client, on):
    """What: the chat names an upload as the person's file was named, even when
    storage keeps it under another name.
    How: two conversations each given a file called rows.csv."""
    names = []
    for _ in range(2):
        upload = SimpleUploadedFile("rows.csv", b"a,y\n1,0\n2,1\n", content_type="text/csv")
        started = json.loads(client.post(reverse("assistant:dataset"), {"dataset_file": upload}).content)
        conversation = Conversation.objects.get(pk=started["conversation"])
        names.append((started["dataset"], conversation.dataset.name))

    assert [shown for shown, _ in names] == ["rows.csv", "rows.csv"]
    assert names[0][1] != names[1][1]


def test_removing_the_dataset_lets_it_go(client, on):
    """What: the picker's × takes the dataset off the conversation, deletes its
    file, and tells the chat.
    How: a CSV uploaded, then the same view posted with `clear`."""
    from pathlib import Path

    upload = SimpleUploadedFile("rows.csv", b"a,y\n1,0\n2,1\n", content_type="text/csv")
    started = json.loads(client.post(reverse("assistant:dataset"), {"dataset_file": upload}).content)
    conversation = Conversation.objects.get(pk=started["conversation"])
    stored = Path(conversation.dataset.path)

    data = json.loads(client.post(reverse("assistant:dataset"),
                                  {"conversation": conversation.pk, "clear": "1"}).content)
    conversation.refresh_from_db()

    assert data["dataset"] == "" and conversation.dataset_path() is None
    assert not stored.exists()
    assert conversation.events.filter(kind="dataset").last().payload == {"name": "", "demo": False}


def test_the_plan_takes_a_reason_for_each_field(on):
    """What: update_plan takes `reasons`, a reason per field by name, and keeps
    them with the plan; a reason that is not text is refused, named by its
    place.
    How: the plan written with two reasons, then with a number for one."""
    conversation = _planning()
    tools = tools_for(conversation)
    ok = tools.execute("update_plan", {"task": "classification", "reasons": {
        "task": "Three named classes.", "model_name": "Quick on small tables."}},
        Context(conversation))
    bad = tools.execute("update_plan", {"reasons": {"task": 3}}, Context(conversation))
    conversation.refresh_from_db()

    assert "Invalid" not in ok.output
    assert conversation.plan["reasons"] == {"task": "Three named classes.",
                                            "model_name": "Quick on small tables."}
    assert "reasons.task must be a string" in bad.output
