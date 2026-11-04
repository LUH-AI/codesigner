"""`Experiment.owner` is `SET_NULL`, and an ownerless experiment is reachable
by nobody. Therefore `manage.py users delete` (`.delete_user`) keeps an account
that owns experiments, and deactivation is the way out.
"""

import io
import json

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from experiments.models import Experiment

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _experiment(owner=None, name="theirs") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner)


def _users(*args):
    out = io.StringIO()
    call_command("users", *args, json=True, stdout=out)
    return json.loads(out.getvalue())


def test_an_orphaned_experiment_is_reachable_by_nobody(hosted, django_user_model):
    """What: an experiment whose owner was deleted is nobody's. How: an
    experiment left ownerless as `SET_NULL` would leave it, asked for by a
    stranger."""
    from access.policy import GroupPolicy

    leaver = django_user_model.objects.create_user(username="leaver", password="pw")
    exp = _experiment(owner=leaver)
    stranger = django_user_model.objects.create_user(username="nosy", password="pw")

    exp.owner = None
    exp.save(update_fields=["owner"])

    request = type("R", (), {"user": stranger})()
    assert not GroupPolicy().experiments(request).filter(pk=exp.pk).exists()


def test_deleting_someone_who_owns_experiments_is_refused(hosted, django_user_model):
    """What: `users delete` keeps an owner and their experiments, and says to
    deactivate instead. How: an account owning one experiment."""
    leaver = django_user_model.objects.create_user("leaver", "leaver@x.de", "pw")
    _experiment(owner=leaver)

    with pytest.raises(CommandError, match="Deactivate it instead"):
        _users("delete", "leaver@x.de")

    assert django_user_model.objects.filter(pk=leaver.pk).exists()
    assert Experiment.objects.get().owner_id == leaver.pk


def test_someone_who_owns_nothing_can_still_be_deleted(hosted, django_user_model):
    """What: the guard protects work only. How: an account owning nothing."""
    nobody = django_user_model.objects.create_user("nobody", "nobody@x.de", "pw")

    _users("delete", "nobody@x.de")

    assert not django_user_model.objects.filter(pk=nobody.pk).exists()


def test_deactivating_blocks_signing_in_and_keeps_the_work(client, hosted, django_user_model):
    """What: `users deactivate` stops sign-in and leaves the work owned. How:
    an owner deactivated, then signing in."""
    leaver = django_user_model.objects.create_user("leaver", "leaver@x.de", "pw")
    _experiment(owner=leaver)

    _users("deactivate", "leaver@x.de")

    assert not client.login(username="leaver", password="pw")
    assert Experiment.objects.get().owner_id == leaver.pk


def test_the_list_shows_how_much_someone_owns(hosted, django_user_model):
    """What: `users list` counts each account's experiments, so an owner shows
    before anyone tries to delete. How: an account owning two."""
    owner = django_user_model.objects.create_user("owner", "owner@x.de", "pw")
    _experiment(owner=owner, name="a")
    _experiment(owner=owner, name="b")

    assert {u["email"]: u["experiments"] for u in _users("list")} == {"owner@x.de": 2}
