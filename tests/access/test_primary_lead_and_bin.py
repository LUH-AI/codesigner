"""Each group has a single primary lead, and the role only moves by transfer.
Deleting a group bins its work for a site admin to settle. A database constraint
and `Group.delete()` enforce both rules outside the views.
"""

import pytest
from django.db import IntegrityError, transaction
from django.urls import reverse

from access.models import Group, Membership
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _person(django_user_model, name, group=None, role=Membership.MEMBER):
    user = django_user_model.objects.create_user(username=name, password="pw")
    if group is not None:
        Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


def _experiment(owner, name, **kw):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, **kw)


@pytest.fixture
def lab(django_user_model):
    group = Group.objects.create(name="lab", user_limit=5)
    spare = Group.objects.create(name="spare", user_limit=5)
    site = _person(django_user_model, "site")
    return {
        "group": group,
        "spare": spare,
        "primary": _person(django_user_model, "vera", group,
                           Membership.PRIMARY_LEAD),
        "lead": _person(django_user_model, "lars", group, Membership.LEAD),
        "member": _person(django_user_model, "ana", group),
        "site": django_user_model.objects.get(pk=site.pk),
    }


def _membership(user):
    return Membership.objects.get(user=user)


def _in_lab(route, lab, who):
    """`_in_lab` addresses a route by group and membership row."""
    return reverse(route, args=[lab["group"].pk, _membership(lab[who]).pk])


def test_an_ordinary_lead_has_no_role_controls(client, hosted, lab):
    """A lead asking for role controls gets a 404."""
    client.force_login(lab["lead"])

    response = client.post(
        _in_lab("ui:group_set_role", lab, "member"),
        {"role": Membership.LEAD})

    assert response.status_code == 404
    assert _membership(lab["member"]).role == Membership.MEMBER


def test_the_primary_lead_makes_somebody_a_lead(client, hosted, lab):
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_set_role", lab, "member"),
                {"role": Membership.LEAD})

    assert _membership(lab["member"]).role == Membership.LEAD


def test_the_primary_lead_cannot_change_their_own_role(client, hosted, lab):
    """Standing down needs a transfer, which names a successor."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_set_role", lab, "primary"),
                {"role": Membership.MEMBER})

    assert _membership(lab["primary"]).role == Membership.PRIMARY_LEAD


def test_the_role_is_not_settable_to_primary(client, hosted, lab):
    """Only a transfer can make a primary lead."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_set_role", lab, "lead"),
                {"role": Membership.PRIMARY_LEAD})

    assert _membership(lab["lead"]).role == Membership.LEAD


def test_an_ordinary_lead_may_only_add_members(client, hosted, lab,
                                               django_user_model):
    """Otherwise one form post would undo the primary role's restriction."""
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                {"identifier": "newbie", "role": Membership.LEAD})

    assert not django_user_model.objects.filter(username="newbie").exists()


def test_the_role_transfers_only_to_a_lead(client, hosted, lab):
    """Promoting and appointing are two decisions."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_transfer_primary", lab, "member"))

    assert _membership(lab["primary"]).role == Membership.PRIMARY_LEAD
    assert _membership(lab["member"]).role == Membership.MEMBER


def test_transferring_leaves_exactly_one_primary_lead(client, hosted, lab):
    """Both rows change together, so the group never has two."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_transfer_primary", lab, "lead"))

    assert _membership(lab["lead"]).role == Membership.PRIMARY_LEAD
    assert _membership(lab["primary"]).role == Membership.LEAD
    assert Membership.objects.filter(group=lab["group"],
                                     role=Membership.PRIMARY_LEAD).count() == 1


def test_a_group_cannot_hold_two_primary_leads(hosted, lab):
    """The admin and data migrations bypass the view."""
    membership = _membership(lab["lead"])
    membership.role = Membership.PRIMARY_LEAD

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            membership.save(update_fields=["role"])


def test_the_primary_lead_cannot_be_removed_from_the_group(client, hosted, lab):
    """The group must keep someone able to appoint a lead."""
    client.force_login(lab["primary"])
    client.post(_in_lab("ui:group_remove_person", lab, "primary"))

    assert Membership.objects.filter(user=lab["primary"]).exists()


def test_a_new_experiment_lands_in_its_owners_group(hosted, lab):
    """The work keeps its group when the owner leaves."""
    exp = _experiment(lab["member"], "theirs")

    assert exp.group_id == lab["group"].pk


def test_an_owner_deleted_leaves_the_work_in_its_group(hosted, lab):
    """A deleted owner leaves nothing to derive a group from."""
    exp = _experiment(lab["member"], "theirs")
    lab["member"].delete()
    exp.refresh_from_db()

    assert exp.owner_id is None
    assert exp.group_id == lab["group"].pk


def _emptied_group_holding_work(django_user_model):
    """`Membership.group` is PROTECT, so only an emptied group can be deleted.
    """
    group = Group.objects.create(name="wound-down", user_limit=2)
    owner = _person(django_user_model, "gone", group)
    exp = _experiment(owner, "left behind")
    Membership.objects.filter(user=owner).delete()
    return group, exp


def test_deleting_a_group_puts_its_work_in_the_bin(hosted, django_user_model):
    group, exp = _emptied_group_holding_work(django_user_model)

    group.delete()
    exp.refresh_from_db()

    assert exp.group_id is None
    assert exp.trashed_at is not None


def test_a_group_with_people_in_it_is_not_deleted(hosted, lab):
    """What: `delete_group` keeps a group that still has people. How: the
    lab group with its three members."""
    from access.services.errors import ServiceError
    from access.services.groups import delete_group

    with pytest.raises(ServiceError):
        delete_group(lab["group"])
    assert Group.objects.filter(pk=lab["group"].pk).exists()


def test_binned_work_is_visible_to_nobody(hosted, lab, django_user_model):
    """Binned work belongs to no group."""
    from access.policy import GroupPolicy

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()

    for user in (lab["primary"], lab["member"], lab["site"]):
        request = type("R", (), {"user": user})()
        names = GroupPolicy().experiments(request).values_list("data__name", flat=True)
        assert "left behind" not in set(names)


def test_the_site_bin_lists_each_experiments_run_count(hosted, lab, django_user_model):
    """What: `trash_rows` lists a deleted group's work with its run count.
    How: an emptied group holding one experiment with one run, deleted."""
    from experiments.services.site import trash_rows

    group, exp = _emptied_group_holding_work(django_user_model)
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 1}, trial_offset=0)
    group.delete()

    assert [e.run_count for e in trash_rows()] == [1]


def test_rehoming_puts_it_back_in_a_group(hosted, lab, django_user_model):
    """What: `rehome` sets the group and clears `trashed_at` together. How: a
    binned experiment rehomed into the lab group."""
    from experiments.services.site import rehome

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()

    rehome(exp, lab["group"])
    exp.refresh_from_db()

    assert exp.group_id == lab["group"].pk
    assert exp.trashed_at is None


def test_a_rehomed_experiment_is_its_new_groups(hosted, lab, django_user_model):
    """What: rehoming makes the work reachable to its new group again. How: a
    binned experiment rehomed, then listed for the group's primary lead."""
    from access.policy import GroupPolicy
    from experiments.services.site import rehome

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    rehome(exp, lab["group"])

    request = type("R", (), {"user": lab["primary"]})()
    names = set(GroupPolicy().experiments(request).values_list("data__name", flat=True))

    assert "left behind" in names


def test_discarding_from_the_bin_is_a_second_decision(hosted, lab, django_user_model):
    """What: deleting a group bins its work, and only `discard` destroys it.
    How: an emptied group deleted, its experiment then discarded."""
    from experiments.services.site import discard

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    assert Experiment.objects.filter(pk=exp.pk).exists()

    discard(Experiment.objects.get(pk=exp.pk))

    assert not Experiment.objects.filter(pk=exp.pk).exists()
