"""The assistant acts as the person, and never as more than them.

Three things are checked here: the policy refuses an assistant's request what
only a person may do, whatever the person may (`AgentLimits`); a tool's request
is the person as they are now — active, signed in, without the permissions that
reach beyond their own work (`Context.request`); and across every person and
experiment of a hosted instance, nothing the assistant may do is something the
person may not.
"""

import pytest
from django.contrib.auth.models import AnonymousUser

from assistant.gateway import AgentRequest, Context
from assistant.models import Conversation
from assistant.tools.registry import Refused
from tests.access.test_groups import hosted, world  # noqa: F401 — fixtures
from access import policy as permissions
from experiments.models import Experiment
from access.policy import ACTIONS, HUMAN_ONLY

pytestmark = pytest.mark.django_db


def _experiment(**fields):
    return Experiment.objects.create(name="e", model_name="Random Forest",
                                     optimizer_name="Random Search", metric_names=["accuracy"],
                                     seed=0, **fields)


class _Person:
    """A person's own request: no assistant."""

    def __init__(self, user):
        self.user = user


# ── the policy narrows an assistant's request ───────────────────────────────

def test_an_assistant_may_not_run_delete_or_share_even_where_anyone_may(settings):
    """What: on an instance without accounts, where everybody may do everything,
    an assistant's request is still refused what only a person may do — and
    nothing else.
    How: every action on one experiment, as a person and as an assistant."""
    settings.ALLOW_CUSTOM_MODELS = True
    exp = _experiment()
    policy = permissions.policy()
    person, agent = _Person(AnonymousUser()), AgentRequest(AnonymousUser())

    for action in ACTIONS:
        assert policy.may(person, exp, action)
        assert policy.may(agent, exp, action) == (action not in HUMAN_ONLY), action
    assert policy.may_upload_models(person) and not policy.may_upload_models(agent)
    assert policy.may_change_defaults(person) and not policy.may_change_defaults(agent)


def test_the_limits_keep_the_policy_what_it_was(settings):
    """What: the policy with the assistant's limits in front is still an
    instance of the configured class, so nothing that asks what it is changes.
    How: the configured policy's type."""
    from access.policy import GroupPolicy

    settings.EXPERIMENT_POLICY = "access.policy.GroupPolicy"
    assert isinstance(permissions.policy(), GroupPolicy)
    assert isinstance(permissions.policy(), permissions.AgentLimits)


def test_nothing_the_assistant_may_do_is_beyond_the_person(hosted, world):  # noqa: F811
    """What: for every person and experiment of a hosted instance, and every
    action, the assistant acting for the person is allowed only what the person
    is, and never what only a person may do.
    How: the whole world fixture, each action asked as the person and as an
    assistant whose request the gateway built."""
    policy = permissions.policy()
    people = ("lead_v", "ana", "ben", "lead_n", "cleo", "site")
    experiments = ("ana_private", "ana_viewer", "ana_contrib", "cleo_private")
    for name in people:
        conversation = Conversation.objects.create(owner=world[name])
        agent = Context(conversation).request()
        for exp_name in experiments:
            exp = world[exp_name]
            for action in ACTIONS:
                as_agent = policy.may(agent, exp, action)
                if as_agent:
                    assert policy.may(_Person(world[name]), exp, action), (name, exp_name, action)
                if action in HUMAN_ONLY:
                    assert not as_agent, (name, exp_name, action)


def test_a_site_admins_reach_is_not_the_assistants(hosted, world, django_user_model):  # noqa: F811
    """What: a superuser sees every experiment themselves; acting for them, the
    assistant sees only what is theirs.
    How: the experiments visible to a superuser, and to an assistant for them."""
    root = django_user_model.objects.create_superuser("root", password="pw")
    policy = permissions.policy()
    agent = Context(Conversation.objects.create(owner=root)).request()

    assert policy.experiments(_Person(root)).count() == 4
    assert policy.experiments(agent).count() == 0


# ── the person, as they are now ─────────────────────────────────────────────

def test_an_assistant_stops_when_its_person_is_deactivated(hosted, world):  # noqa: F811
    """What: a person who is no longer active cannot be acted for.
    How: deactivates ana, then asks for a request for her conversation."""
    conversation = Conversation.objects.create(owner=world["ana"])
    world["ana"].is_active = False
    world["ana"].save(update_fields=["is_active"])

    with pytest.raises(Refused):
        Context(conversation).request()


def test_an_assistant_acts_only_while_its_browser_is_signed_in(client, hosted, world):  # noqa: F811
    """What: a conversation started from a browser acts only while that session
    is signed in as its owner — not after signing out, nor after the password
    changes.
    How: a real session, used, then the password changed, then signed out."""
    ana = world["ana"]
    client.force_login(ana)
    conversation = Conversation.objects.create(owner=ana, session_key=client.session.session_key)

    assert Context(conversation).request().user.pk == ana.pk

    ana.set_password("another")
    ana.save()
    with pytest.raises(Refused):
        Context(conversation).request()

    client.force_login(ana)
    conversation.session_key = client.session.session_key
    conversation.save(update_fields=["session_key"])
    client.logout()
    with pytest.raises(Refused):
        Context(conversation).request()


def test_an_experiment_is_reached_only_as_the_views_reach_it(hosted, world):  # noqa: F811
    """What: through the gateway, ben reaches what is shared with him, not what
    is not, and nothing for a human-only action.
    How: `Context.experiment` for ben on a shared and a private experiment."""
    context = Context(Conversation.objects.create(owner=world["ben"]))

    assert context.experiment(world["ana_contrib"].pk, permissions.EDIT).pk == world["ana_contrib"].pk
    with pytest.raises(Refused):
        context.experiment(world["ana_private"].pk, permissions.VIEW)
    with pytest.raises(Refused):
        context.experiment(world["ana_contrib"].pk, permissions.DELETE)


def test_tool_code_never_looks_experiments_up_itself():
    """What: no tool finds an experiment except through the gateway, so none can
    reach one the person's pages could not.
    How: the source of every module under assistant/tools/."""
    from pathlib import Path

    import assistant

    for path in (Path(assistant.__file__).parent / "tools").glob("*.py"):
        source = path.read_text()
        assert "Experiment.objects" not in source, path.name
        assert "ExperimentData.objects" not in source, path.name
