"""`seed_demo` builds groups, roles and all three sharing levels, and must stay
re-runnable. The command stops with an error when DEBUG is off, since every
password it sets is the username.
"""

import pytest
from django.core.management import CommandError, call_command

from access.models import Group, Membership
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db


@pytest.fixture
def debugging(settings):
    settings.DEBUG = True
    return settings


def test_it_builds_the_whole_shape(debugging):
    call_command("seed_demo", verbosity=0)

    assert set(Group.objects.values_list("name", flat=True)) == {
        "site", "vision-lab", "nlp-group"}
    # Only a primary lead can appoint other leads.
    assert Membership.objects.filter(role=Membership.PRIMARY_LEAD).count() == 3
    assert Run.objects.filter(status="running").exists(), \
        "the Jobs panel needs something to stop"


def test_all_three_sharing_levels_are_represented(debugging):
    """The interesting cases are the boundaries between levels."""
    call_command("seed_demo", verbosity=0)

    from experiments.models import ExperimentShare

    assert Experiment.objects.filter(shares__level=ExperimentShare.VIEWER).exists()
    assert Experiment.objects.filter(shares__level=ExperimentShare.CONTRIBUTOR).exists()
    assert Experiment.objects.filter(shares__isnull=True).exists()


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
    real = Group.objects.create(name="a-real-group", user_limit=4)
    theirs = django_user_model.objects.create_user("someone", password="pw")
    Membership.objects.create(user=theirs, group=real)

    call_command("seed_demo", reset=True, verbosity=0)

    assert Group.objects.filter(pk=real.pk).exists()
    assert Membership.objects.filter(user=theirs).exists()


def test_it_refuses_with_debug_off(settings):
    """Known passwords on a real host are the risk the refusal guards."""
    settings.DEBUG = False

    with pytest.raises(CommandError, match="DEBUG is off"):
        call_command("seed_demo", verbosity=0)

    assert not Group.objects.exists()


def test_force_is_the_way_past_that(settings):
    settings.DEBUG = False

    call_command("seed_demo", force=True, verbosity=0)

    assert Group.objects.exists()
