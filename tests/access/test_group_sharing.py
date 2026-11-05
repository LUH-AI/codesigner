"""Sharing an experiment with its whole group (`Experiment.group_share`): who
reaches it at which level, people who join later, the bin, and who may set
it."""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from access.policy import DELETE, RUN, VIEW, GroupPolicy, share_of
from experiments.models import Experiment, ExperimentShare
from experiments.services import bin as bin_service

VIEWER, CONTRIBUTOR = ExperimentShare.VIEWER, ExperimentShare.CONTRIBUTOR

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


@pytest.fixture
def lab(django_user_model):
    """`lab` is a group with an owner and two colleagues, and an outsider in
    another group; the owner has one experiment."""
    vision, other = (Group.objects.create(name=n, user_limit=9) for n in ("vision", "other"))
    people = {}
    for name, group in (("ana", vision), ("ben", vision), ("cleo", vision), ("dan", other)):
        people[name] = django_user_model.objects.create_user(name, password="pw")
        Membership.objects.create(user=people[name], group=group)
    exp = Experiment.objects.create(
        name="study", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=people["ana"], group=vision)
    return {"group": vision, "exp": exp, **people}


def _request(user):
    return type("R", (), {"user": user})()


def test_a_group_viewer_share_reaches_every_member_read_only(hosted, lab):
    """What: a group-wide viewer share lets every member of the group see the
    experiment but not run it, and nobody outside. How: share at viewer, then
    ask `GroupPolicy` for each person."""
    lab["exp"].group_share = VIEWER
    lab["exp"].save()
    policy = GroupPolicy()

    for name in ("ben", "cleo"):
        assert lab["exp"] in policy.experiments(_request(lab[name]))
        assert policy.may(_request(lab[name]), lab["exp"], VIEW)
        assert not policy.may(_request(lab[name]), lab["exp"], RUN)
    assert lab["exp"] not in policy.experiments(_request(lab["dan"]))


def test_a_group_contributor_share_lets_members_run_and_delete(hosted, lab):
    """What: a group-wide contributor share lets members run and delete, and
    a deletion lands in every member's bin. How: share at contributor, check
    `may`, then delete."""
    lab["exp"].group_share = CONTRIBUTOR
    lab["exp"].save()

    assert GroupPolicy().may(_request(lab["cleo"]), lab["exp"], DELETE)
    bin_service.delete(lab["exp"], by=lab["cleo"])
    assert set(lab["exp"].bin_entries.values_list("user__username", flat=True)) == {
        "ana", "ben", "cleo"}


def test_people_who_join_later_are_included(hosted, lab, django_user_model):
    """What: the group-wide share covers a person added after it was set. How:
    share, then add a newcomer to the group."""
    lab["exp"].group_share = VIEWER
    lab["exp"].save()
    newcomer = django_user_model.objects.create_user("eve", password="pw")
    Membership.objects.create(user=newcomer, group=lab["group"])

    assert share_of(newcomer, lab["exp"]) == VIEWER


def test_the_stronger_of_own_and_group_share_counts(hosted, lab):
    """What: `share_of` returns the stronger level when a person has their own
    share and the group has another. How: ben a contributor, the group
    viewers."""
    ExperimentShare.objects.create(experiment=lab["exp"], user=lab["ben"], level=CONTRIBUTOR)
    lab["exp"].group_share = VIEWER
    lab["exp"].save()

    assert share_of(lab["ben"], lab["exp"]) == CONTRIBUTOR
    assert share_of(lab["cleo"], lab["exp"]) == VIEWER


def test_the_owner_sets_it_on_the_group_page(client, hosted, lab):
    """What: the owner's "Everyone in" selector sets the group-wide share, the
    group page then lists the experiment under "Shared with you" for a
    colleague, and a colleague cannot set it. How: posts by ana and ben."""
    url = reverse("ui:experiment_share_group", args=[lab["exp"].pk])
    client.force_login(lab["ana"])
    assert "Everyone in vision" in client.get(
        reverse("ui:group_detail", args=[lab["group"].pk])).content.decode()
    client.post(url, {"level": VIEWER})
    lab["exp"].refresh_from_db()
    assert lab["exp"].group_share == VIEWER

    client.force_login(lab["ben"])
    page = client.get(reverse("ui:group_detail", args=[lab["group"].pk]))
    assert [e.pk for e in page.context["shared"]] == [lab["exp"].pk]
    assert page.context["shared"][0].level == VIEWER
    assert client.post(url, {"level": "none"}).status_code in (403, 404)
    lab["exp"].refresh_from_db()
    assert lab["exp"].group_share == VIEWER
