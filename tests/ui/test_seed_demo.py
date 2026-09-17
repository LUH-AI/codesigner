"""The worked example, and that it stays safe to re-run.

*What:* `seed_demo` builds an instance with groups, roles and all three sharing
levels in it, so the panels can be looked at without setting that up by hand
every time. Its value depends entirely on being re-runnable, and its danger is
that every password it sets is the username.

*How:* run it twice and assert nothing doubled; run `--reset` and assert it
rebuilt; and assert it refuses outright with DEBUG off, which is the guard
between a debugging convenience and a set of known passwords on a real host.
"""

import pytest
from django.core.management import CommandError, call_command

from access.models import Group, Membership
from ui.models import Experiment, Run

pytestmark = pytest.mark.django_db


@pytest.fixture
def debugging(settings):
    settings.DEBUG = True
    return settings


def test_it_builds_the_whole_shape(debugging):
    call_command("seed_demo", verbosity=0)

    assert set(Group.objects.values_list("name", flat=True)) == {
        "site", "vision-lab", "nlp-group"}
    assert Membership.objects.filter(role=Membership.LEAD).count() == 3
    assert Run.objects.filter(status="running").exists(), \
        "the Jobs panel needs something to stop"


def test_all_three_sharing_levels_are_represented(debugging):
    """The interesting cases are the boundaries between them, so an example
    with only the easy one would not be worth having."""
    call_command("seed_demo", verbosity=0)

    assert Experiment.objects.filter(shared=True).exists()
    assert Experiment.objects.filter(shared_with__isnull=False).exists()
    assert Experiment.objects.filter(shared=False, shared_with__isnull=True).exists()


def test_running_it_twice_changes_nothing(debugging):
    call_command("seed_demo", verbosity=0)
    before = (Group.objects.count(), Membership.objects.count(),
              Experiment.objects.count(), Run.objects.count())

    call_command("seed_demo", verbosity=0)

    assert (Group.objects.count(), Membership.objects.count(),
            Experiment.objects.count(), Run.objects.count()) == before


def test_reset_rebuilds_rather_than_accumulating(debugging):
    call_command("seed_demo", verbosity=0)
    call_command("seed_demo", reset=True, verbosity=0)

    assert Group.objects.count() == 3


def test_reset_leaves_anything_it_did_not_make(debugging, django_user_model):
    """A demo that deleted an operator's real group would be a worse problem
    than the one it solves."""
    real = Group.objects.create(name="a-real-group", user_limit=4)
    theirs = django_user_model.objects.create_user("someone", password="pw")
    Membership.objects.create(user=theirs, group=real)

    call_command("seed_demo", reset=True, verbosity=0)

    assert Group.objects.filter(pk=real.pk).exists()
    assert Membership.objects.filter(user=theirs).exists()


def test_it_refuses_with_debug_off(settings):
    """Every password here is the username. That is fine on a laptop and is the
    kind of thing that ends up on a real host by accident, so the command is
    the thing that says no."""
    settings.DEBUG = False

    with pytest.raises(CommandError, match="DEBUG is off"):
        call_command("seed_demo", verbosity=0)

    assert not Group.objects.exists()


def test_force_is_the_way_past_that(settings):
    settings.DEBUG = False

    call_command("seed_demo", force=True, verbosity=0)

    assert Group.objects.exists()
