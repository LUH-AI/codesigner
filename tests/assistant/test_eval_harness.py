"""The evals' harness (tests/assistant/evals), checked with a scripted model:
situations are built as the person would find them, the stand-in page does what
act.js would, and runs are scored as the cases say."""

import pytest

from tests.assistant.evals.harness import Case, build, load_cases, matches, run, score
from tests.assistant.helpers import MockLLM, call, text

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def assistant_on(settings):
    settings.ASSISTANT_ENABLED = True


def _case(situation, expect, say="Hello"):
    return Case(id="c", group="g", say=[say], situation=situation, expect=expect)


def test_a_case_whose_expectations_are_met_passes(client):
    """What: a run that makes the expected calls in order and ends as expected
    passes. How: the scripted model profiles the iris demo and then asks a
    question with choices while planning."""
    llm = MockLLM(call("profile_dataset"),
                  call("present_choices", question="Predict which column?",
                       options=["variety", "petal.width"]))
    case = _case({"phase": "planning", "dataset": "iris"},
                 {"calls": [{"tool": "profile_dataset"},
                            {"tool": "present_choices", "args": {"options": ["variety", "petal.width"]}}],
                  "never": ["update_plan"], "ended": ["waiting"]})

    outcome = run(case, llm, client)

    assert outcome.passed, outcome.failures
    assert [c["tool"] for c in outcome.calls] == ["profile_dataset", "present_choices"]


def test_a_case_falling_short_says_how(client):
    """What: a run that skips the expected call and claims a save fails with a
    reason for each. How: the scripted model answers in words at once on a
    draft's first step."""
    llm = MockLLM(text("Done, I saved ten folds."))
    case = _case({"phase": "setup", "dataset": "iris", "plan": "iris"},
                 {"calls": [{"tool": "fill"}], "absent": ["saved"], "ended": ["answered"]})

    outcome = run(case, llm, client)

    assert not outcome.passed
    assert outcome.failures == ["no fill after call 0", "answer says 'saved'"]


def test_a_draft_opens_filled_from_the_plan_and_fill_changes_the_page(client, monkeypatch):
    """What: the setup situation lands on step 1 filled from the approved plan,
    and the assistant's fill changes the stand-in page. How: the scripted
    model looks at the page, then fills ten folds; the test reads the page's
    values after the run."""
    llm = MockLLM(call("look_at_page"), call("fill", values={"evaluation_value": 10},
                                             reason="More folds for a steadier estimate."),
                  text("Set ten folds; save the step when ready."))
    case = _case({"phase": "setup", "dataset": "iris", "plan": "iris"},
                 {"calls": [{"tool": "look_at_page"},
                            {"tool": "fill", "args": {"values": {"evaluation_value": 10}}}],
                  "mentions": ["save"]})
    pages = []

    def kept(situation, client):
        conversation, page = build(situation, client)
        pages.append(page)
        return conversation, page

    monkeypatch.setattr("tests.assistant.evals.harness.build", kept)
    outcome = run(case, llm, client)

    assert outcome.passed, outcome.failures
    page = pages[0]
    assert page.value("model_name") == "Random Forest"
    assert page.value("evaluation_value") == "10"
    assert page.log[-1][:3] == ("fill", {"values": {"evaluation_value": 10},
                                         "reason": "More folds for a steadier estimate."}, True)


def test_the_dashboard_situation_is_a_created_experiment_with_the_run_form_filled(client):
    """What: the dashboard situation saves the plan's steps, creates the
    experiment and opens its page with the run form filled from the plan. How:
    the test builds it and reads the stand-in page."""
    conversation, page = build({"phase": "dashboard", "dataset": "iris", "plan": "iris"}, client)

    assert conversation.phase == "dashboard"
    assert not conversation.experiment.draft
    assert page.value("optimize_metric") == "accuracy"
    assert page.value("max_trials") == "30"
    assert any(t["name"] == "run" for t in page.targets)


def test_matching_ignores_case_and_reads_tables_key_by_key():
    """What: expected arguments match strings ignoring case and tables by the
    keys they give. How: `matches` and `score` on hand-made calls."""
    assert matches({"values": {"model_name": "svm"}}, {"values": {"model_name": "SVM", "seed": 1}})
    assert not matches({"values": {"model_name": "svm"}}, {"values": {}})
    assert matches(10, 10.0)
    calls = [{"tool": "fill", "arguments": {}}, {"tool": "explain_how", "arguments": {"action": "run"}}]
    assert score({"calls": [{"tool": "explain_how", "args": {"action": "run"}}, {"tool": "fill"}]},
                 calls, "", "answered") == ["no fill after call 2"]
    assert score({"never": ["fill"], "max_calls": 1}, calls, "", "") == ["called fill",
                                                                         "2 calls, more than 1"]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: f"{c.group}/{c.id}")
def test_every_case_situation_builds(case, client):
    """What: each eval case's situation can be set up, with its page open. How:
    `build` runs for the case without a model, and the page must have
    rendered."""
    conversation, page = build(case.situation, client)

    assert page.status == 200, page.path
    assert conversation.browser == page.browser
    assert page.fields or page.targets, page.path
