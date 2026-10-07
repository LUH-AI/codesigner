"""Which group a new experiment goes in, for somebody in more than one.

*What:* somebody in one group is never asked — the experiment is filed there.
Somebody in two or more is asked on every page that creates one: the create
form and both imports. The answer is required, and only one of their own groups
is accepted, because a guess or a stranger's group would file work under a
boundary nobody chose.

*How:* through the pages, as a person in one group and as a person in two.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment

from tests.conftest import DATASETS_DIR, FIXTURES_DIR, get_new_experiment, post_new_experiment


def _valid_post(**overrides):
    data = {"name": "my-exp", "model_name": "Random Forest",
            "optimizer_name": "Random Search",
            "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification", "seed": 0}
    data.update(overrides)
    return data


@pytest.fixture
def people(settings, django_user_model):
    settings.REQUIRE_LOGIN = True
    a = Group.objects.create(name="a", user_limit=5)
    b = Group.objects.create(name="b", user_limit=5)
    c = Group.objects.create(name="c", user_limit=5)
    one = django_user_model.objects.create_user(username="one")
    two = django_user_model.objects.create_user(username="two")
    Membership.objects.create(user=one, group=a)
    Membership.objects.create(user=two, group=a)
    Membership.objects.create(user=two, group=b)
    return {"a": a, "b": b, "c": c, "one": one, "two": two}


def test_somebody_in_one_group_is_not_asked(client, people):
    client.force_login(people["one"])

    assert "group" not in get_new_experiment(client).context["form"].fields
    post_new_experiment(client, _valid_post())

    assert Experiment.objects.get(draft=False).group == people["a"]


def test_somebody_in_two_is(client, people):
    client.force_login(people["two"])

    field = get_new_experiment(client).context["form"].fields["group"]

    assert [label for _, label in field.choices][1:] == ["a", "b"]


def test_and_their_answer_is_where_it_goes(client, people):
    client.force_login(people["two"])

    post_new_experiment(client, _valid_post(group=people["b"].pk))

    assert Experiment.objects.get(draft=False).group == people["b"]


def test_no_answer_creates_nothing(client, people):
    client.force_login(people["two"])

    resp = post_new_experiment(client, _valid_post())

    assert resp.status_code == 200
    assert "group" in resp.context["form"].errors
    assert not Experiment.objects.filter(draft=False).exists()


def test_a_group_they_are_not_in_is_refused(client, people):
    client.force_login(people["two"])

    resp = post_new_experiment(client, _valid_post(group=people["c"].pk))

    assert "group" in resp.context["form"].errors
    assert not Experiment.objects.filter(draft=False).exists()


def test_importing_asks_too(client, people):
    client.force_login(people["two"])
    ihpo = FIXTURES_DIR / "test2.ihpo"

    with open(ihpo, "rb") as f:
        refused = client.post(reverse("ui:import_experiment"), {"file": f})
    with open(ihpo, "rb") as f:
        client.post(reverse("ui:import_experiment"), {"file": f, "group": people["b"].pk})

    assert "Choose which group" in refused.content.decode()
    assert Experiment.objects.get(draft=False).group == people["b"]


def test_the_import_page_offers_only_their_groups(client, people):
    client.force_login(people["two"])

    assert client.get(reverse("ui:import_experiment")).context["groups"] == [
        people["a"], people["b"]]
