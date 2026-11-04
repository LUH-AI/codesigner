"""The site-management commands (`groups`, `members`, `users`, `trash`, `jobs`,
`usage`, `defaults`): each subcommand does what it says through the shared
services, prints JSON a script can read, and reports a refusal as a
`CommandError`.
"""

import io
import json

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from access.models import Group, Membership
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db


def run(*args, **options):
    """`run` calls a command with `--json` and returns what it printed, read."""
    out = io.StringIO()
    call_command(*args, json=True, stdout=out, **options)
    text = out.getvalue()
    return json.loads(text) if text.strip() else None


def test_groups_are_created_listed_updated_and_deleted():
    """What: `groups create/list/update/delete` round-trip a group. How: a
    group with a lead, renamed and resized, then emptied and deleted."""
    made = run("groups", "create", "vision", "--seats", "3", "--lead", "vera")
    assert made["name"] == "vision" and made["lead"] == "vera"
    assert "first time they sign in" in made["message"]
    assert [g["name"] for g in run("groups", "list")] == ["vision"]

    changed = run("groups", "update", "vision", "--name", "vision-lab", "--seats", "5",
                  "--storage-gb", "2", "--inactive")
    assert (changed["name"], changed["seats"], changed["active"]) == ("vision-lab", 5, False)
    assert changed["storage_limit_bytes"] == 2 * 1024 ** 3

    with pytest.raises(CommandError, match="still has people"):
        run("groups", "delete", "vision-lab")
    Membership.objects.all().delete()
    assert run("groups", "delete", "vision-lab") == {"deleted": "vision-lab"}
    assert not Group.objects.exists()


def test_members_are_added_listed_promoted_handed_the_lead_and_removed():
    """What: `members add/list/set-role/transfer/remove` manage a group's people.
    How: a lead and a newcomer through every subcommand."""
    run("groups", "create", "vision", "--seats", "3", "--lead", "vera")
    added = run("members", "add", "vision", "ana")
    assert added["created"] and added["role"] == Membership.MEMBER
    with pytest.raises(CommandError, match="already in"):
        run("members", "add", "vision", "ANA")

    assert run("members", "set-role", "vision", "ana", "lead")["role"] == Membership.LEAD
    assert run("members", "transfer", "vision", "ana")["role"] == Membership.PRIMARY_LEAD
    roles = {m["username"]: m["role"] for m in run("members", "list", "vision")}
    assert roles == {"ana": Membership.PRIMARY_LEAD, "vera": Membership.LEAD}

    assert run("members", "remove", "vision", "vera")["removed"] == "vera"
    with pytest.raises(CommandError, match="is not in"):
        run("members", "remove", "vision", "vera")


def test_users_are_listed_deactivated_and_granted_custom_models(django_user_model):
    """What: `users list/deactivate/activate/custom-models/delete` manage an
    account. How: one account through each subcommand."""
    django_user_model.objects.create_user("ana", "ana@lab.de", "pw")
    assert [u["email"] for u in run("users", "list")] == ["ana@lab.de"]
    assert run("users", "deactivate", "ana@lab.de")["active"] is False
    assert run("users", "activate", "ana@lab.de")["active"] is True
    assert run("users", "custom-models", "ana@lab.de", "on")["custom_models"] is True
    assert run("users", "delete", "ana@lab.de") == {"deleted": "ana@lab.de"}
    with pytest.raises(CommandError, match="no account"):
        run("users", "delete", "ana@lab.de")


def test_the_site_bin_is_listed_rehomed_and_discarded(django_user_model):
    """What: `trash list/rehome/discard` work the site bin by identifier. How:
    two experiments of a deleted group, one rehomed, one discarded."""
    owner = django_user_model.objects.create_user("ana", "ana@lab.de", "pw")
    gone = Group.objects.create(name="gone", user_limit=1)
    kept, dropped = (Experiment.objects.create(
        name=n, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, group=gone) for n in "ab")
    gone.delete()
    home = Group.objects.create(name="home", user_limit=1)

    assert {e["identifier"] for e in run("trash", "list")} == {kept.identifier, dropped.identifier}
    run("trash", "rehome", kept.identifier, "home")
    run("trash", "discard", dropped.identifier)

    assert Experiment.objects.get(pk=kept.pk).group == home
    assert not Experiment.objects.filter(pk=dropped.pk).exists()
    with pytest.raises(CommandError, match="not in the site bin"):
        run("trash", "discard", dropped.identifier)


def test_jobs_are_listed_and_stopped(django_user_model):
    """What: `jobs list/stop` show active runs by identifier and stop one with
    a reason. How: one running run, stopped."""
    owner = django_user_model.objects.create_user("ana", "ana@lab.de", "pw")
    exp = Experiment.objects.create(name="x", model_name="Random Forest",
                                    optimizer_name="Random Search", metric_names=["accuracy"],
                                    seed=0, owner=owner)
    running = Run.objects.create(experiment=exp, status="running", primary_metric="accuracy")

    listed = run("jobs", "list")
    assert [j["identifier"] for j in listed] == [exp.identifier]
    run("jobs", "stop", str(running.pk), "--reason", "maintenance")

    running.refresh_from_db()
    assert running.status == "error" and running.error == "maintenance"
    with pytest.raises(CommandError, match="not in progress"):
        run("jobs", "stop", str(running.pk))


def test_usage_reports_groups_and_people_as_json_and_csv():
    """What: `usage groups/people` report totals as JSON and as CSV. How: one
    group with its lead, in both forms."""
    run("groups", "create", "vision", "--seats", "3", "--lead", "vera@lab.de")
    assert run("usage", "groups")[0] == {"group": "vision", "members": 1, "runs": 0,
                                         "trials": 0, "trial_seconds": 0}
    out = io.StringIO()
    call_command("usage", "people", csv=True, stdout=out)
    assert out.getvalue().splitlines()[0] == "group,email,role,runs,trials,trial_seconds"


def test_defaults_are_shown_set_and_reset():
    """What: `defaults show/set/reset` read and change the inherited settings,
    coerced like the settings form, and refuse an unknown key. How: one
    integer setting changed, then the defaults reset."""
    shown = run("defaults", "show")
    key = next(k for k, v in shown.items() if isinstance(v, int) and not isinstance(v, bool))
    changed = run("defaults", "set", f"{key}={shown[key] + 1}")
    assert changed[key] == shown[key] + 1
    with pytest.raises(CommandError, match="known setting"):
        run("defaults", "set", "no_such_setting=1")
    assert run("defaults", "reset") == shown
