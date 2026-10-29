"""The evals: scripted conversations with a real model, scored on what the
assistant did and said.

A case is a table in a TOML file of tests/assistant/evals/cases/, and the
file's name is the case's group:

    [[case]]
    id = "profile-before-claims"
    say = ["Which column should I predict?"]
    [case.situation]
    phase = "planning"            # planning, setup, dashboard or results
    dataset = "iris"              # a demo dataset, or
    dataset_file = "hostile.csv"  # a file of tests/assistant/evals/data, uploaded
    plan = "iris"                 # a stock plan from PLANS, or a table
    page = "ui:home"              # the page open, by URL name
    [case.expect]
    calls = [{tool = "profile_dataset"}, {tool = "present_choices"}]
    never = ["update_plan"]
    mentions = ["variety"]
    absent = ["saved"]
    ended = ["waiting"]
    plan = {task = "classification"}
    plan_ready = true

`calls` must appear in that order among the calls made, each matched on the
arguments it gives (a string ignoring case, a table key by key). `never`
lists tools that must not be called, `mentions` and `absent` words that the
last turn's answer must and must not contain, and `ended` how the last turn
may end (assistant/loop.py `run_turn`). `plan` is matched against the plan
as the run leaves it, and `plan_ready` asks that the setup's own forms would
take it (`assistant.plan.problems`).
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from django.urls import reverse

CASES_DIR = Path(__file__).parent / "cases"
DATA_DIR = Path(__file__).parent / "data"

#: Plans a situation can start from by name, as the planning stage would leave them.
PLANS = {
    "iris": {
        "goal": "Predict the iris variety from the flower's measurements.",
        "name": "Iris varieties",
        "task": "classification",
        "target_column": "variety",
        "model_name": "Random Forest",
        "evaluation": {"scheme": "kfold", "value": 5},
        "optimizer": {"name": "SMAC"},
        "metric": "accuracy",
        "stopping": {"max_trials": 30},
        "reasons": {"model_name": "A random forest needs no scaling and handles few rows well.",
                    "evaluation": "Five folds suit 150 rows.",
                    "stopping": "Thirty trials cover a forest's few hyperparameters."},
    },
}


@dataclass
class Case:
    """`Case` is one scripted conversation and what is expected of it."""

    id: str
    group: str
    say: list[str]
    situation: dict
    expect: dict


@dataclass
class Outcome:
    """`Outcome` is how one run of a case went."""

    case: str
    group: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    answer: str = ""
    ended: str = ""
    plan: dict = field(default_factory=dict)
    tokens: int = 0
    seconds: float = 0.0


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    """`load_cases` reads every case under *directory*, in file and table order."""
    cases = []
    for path in sorted(directory.glob("*.toml")):
        for table in tomllib.loads(path.read_text(encoding="utf-8")).get("case", []):
            say = table["say"]
            cases.append(Case(id=table["id"], group=path.stem,
                              say=[say] if isinstance(say, str) else list(say),
                              situation=dict(table.get("situation", {})),
                              expect=dict(table.get("expect", {}))))
    return cases


def _request():
    from django.contrib.auth.models import AnonymousUser
    from django.test import RequestFactory

    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    return request


def build(situation: dict, client):
    """`build` sets up *situation* on an instance without accounts, with
    *client* as the person's browser, and returns `(conversation, page)`. The
    page is open where the situation says."""
    from assistant.models import Conversation
    from experiments.services import snapshot
    from ui import drafts, setup_steps as steps

    from assistant import plan as plans
    from core import io

    from .page import SimulatedPage

    client.get("/")
    browser = client.session.session_key
    if browser is None:
        client.session.save()
        browser = client.session.session_key
    phase = situation.get("phase", "planning")
    plan = situation.get("plan") or {}
    if isinstance(plan, str):
        plan = dict(PLANS[plan])
    page = SimulatedPage(client, browser)

    if phase == "results":
        from tests.conftest import FIXTURES_DIR

        exp = snapshot.experiment_from_snapshot(
            io.parse((FIXTURES_DIR / f"{situation.get('fixture', 'analytics')}.ihpo").read_bytes()))
        conversation = Conversation.objects.create(browser=browser, experiment=exp,
                                                   phase=Conversation.DASHBOARD)
        page.open(reverse(situation.get("page", "ui:experiment_detail"), args=[exp.pk]))
        return conversation, page

    conversation = Conversation.objects.create(browser=browser, plan=plan,
                                               demo_dataset=situation.get("dataset", ""))
    if situation.get("dataset_file"):
        from django.core.files.base import ContentFile

        name = situation["dataset_file"]
        conversation.dataset.save(name, ContentFile((DATA_DIR / name).read_bytes()), save=False)
        conversation.dataset_filename = name
        conversation.save()
    if phase == "planning":
        page.open(reverse(situation.get("page", "ui:home")))
        return conversation, page

    response = client.post(reverse("assistant:approve"), {"conversation": conversation.pk})
    assert response.status_code == 200, response.content
    conversation.refresh_from_db()
    exp = conversation.experiment
    saved = situation.get("saved", [] if phase == "setup" else list(steps.STEPS[:3]))
    forms = {steps.SETUP: plans.setup_data(plan), steps.OPTIMIZER: plans.optimizer_data(plan),
             steps.DATA: plans.processing_data(plan)}
    request = _request()
    for step in saved:
        outcome = drafts.save_step(request, exp, step, forms[step])
        assert outcome.saved, (step, outcome.refusal, outcome.form and outcome.form.errors)
    if phase == "dashboard":
        problem = drafts.create(request, exp)
        assert problem is None, problem
        conversation.phase = Conversation.DASHBOARD
        conversation.save(update_fields=["phase"])
        page.open(reverse(situation.get("page", "ui:experiment_detail"), args=[exp.pk]))
    else:
        page.open(reverse(situation.get("page", "ui:setup"), args=[exp.pk]))
    return conversation, page


def run(case: Case, llm, client) -> Outcome:
    """`run` plays *case* against *llm* and scores what happened."""
    from assistant.loop import run_turn
    from assistant.models import AgentAction, Message
    from assistant.turns import tools_for

    started = time.monotonic()
    conversation, page = build(case.situation, client)
    ended = ""
    with mock.patch("assistant.acts.wait", lambda act, seconds: (page.carry_out(act), act)[1]):
        for text in case.say:
            conversation.refresh_from_db()
            ended = run_turn(conversation, text, llm=llm, tools=tools_for(conversation), stream=False)
    calls = [{"tool": a.tool, "arguments": a.arguments, "outcome": a.outcome, "detail": a.detail[:300]}
             for a in AgentAction.objects.filter(conversation=conversation).order_by("id")]
    messages = list(conversation.messages.order_by("id"))
    last_user = max((i for i, m in enumerate(messages) if m.role == Message.USER), default=-1)
    answer = "\n".join(m.content for m in messages[last_user + 1:]
                       if m.role == Message.ASSISTANT and m.content)
    conversation.refresh_from_db()
    failures = score(case.expect, calls, answer, ended)
    if "plan" in case.expect and not matches(case.expect["plan"], conversation.plan):
        failures.append(f"plan is {conversation.plan}")
    if case.expect.get("plan_ready"):
        from assistant.plan import problems

        found = problems(conversation.plan, _request(), conversation.dataset_path())
        if found:
            failures.append(f"plan not ready: {found}")
    return Outcome(case=case.id, group=case.group, passed=not failures, failures=failures,
                   calls=calls, answer=answer, ended=ended, plan=conversation.plan,
                   tokens=sum(m.tokens for m in messages),
                   seconds=round(time.monotonic() - started, 1))


def matches(expected, actual) -> bool:
    """`matches` tells whether *actual* gives what *expected* asks for:
    strings equal ignoring case, tables key by key, anything else equal."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(v, actual[k])
                                                for k, v in expected.items())
    if isinstance(expected, str):
        return isinstance(actual, str) and expected.casefold() == actual.casefold()
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 1e-9
        except (TypeError, ValueError):
            return False
    return expected == actual


def score(expect: dict, calls: list[dict], answer: str, ended: str) -> list[str]:
    """`score` returns why a run falls short of *expect*, one reason each."""
    failures = []
    position = 0
    for wanted in expect.get("calls", []):
        found = next((i for i in range(position, len(calls))
                      if calls[i]["tool"] == wanted["tool"]
                      and matches(wanted.get("args", {}), calls[i]["arguments"])), None)
        if found is None:
            failures.append(f"no {wanted['tool']}{wanted.get('args', '')} after call {position}")
        else:
            position = found + 1
    made = {c["tool"] for c in calls}
    failures += [f"called {tool}" for tool in expect.get("never", []) if tool in made]
    text = answer.casefold()
    failures += [f"answer lacks {word!r}" for word in expect.get("mentions", [])
                 if word.casefold() not in text]
    failures += [f"answer says {word!r}" for word in expect.get("absent", [])
                 if word.casefold() in text]
    if expect.get("ended") and ended not in expect["ended"]:
        failures.append(f"turn ended {ended!r}, not {expect['ended']}")
    if "max_calls" in expect and len(calls) > expect["max_calls"]:
        failures.append(f"{len(calls)} calls, more than {expect['max_calls']}")
    return failures
