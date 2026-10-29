"""Each eval case, played against the configured model (`--eval`)."""

from dataclasses import asdict

import pytest

from .harness import run

pytestmark = pytest.mark.django_db


@pytest.fixture
def live_model(settings):
    """`live_model` turns the assistant on, as a developer's `.env` has it,
    against the model its settings name."""
    from assistant.llm import ChatClient

    settings.ASSISTANT_ENABLED = True
    return ChatClient.from_settings()


def test_case(case, repeat, live_model, client, eval_report):
    """What: the assistant does what the case expects. How: the case's
    situation is built, its messages played as turns against the real model,
    and the calls and the last answer scored."""
    outcome = run(case, live_model, client)
    eval_report.append(asdict(outcome))
    assert outcome.passed, "\n".join(outcome.failures + [f"calls: {[c['tool'] for c in outcome.calls]}",
                                                         f"answer: {outcome.answer[:500]}"])
