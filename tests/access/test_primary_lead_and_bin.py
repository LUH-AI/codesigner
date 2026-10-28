"""Each group has a single primary lead, and the role moves only by transfer.
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
    from django.contrib.auth.models import Permission

    group = Group.objects.create(name="lab", user_limit=5)
    spare = Group.objects.create(name="spare", user_limit=5)
    site = _person(django_user_model, "site")
    site.user_permissions.add(Permission.objects.get(
        content_type__app_label="access", codename="manage_site"))
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
                {"email": "newbie@example.org", "role": Membership.LEAD})

    assert not django_user_model.objects.filter(email="newbie@example.org").exists()


def test_the_role_transfers_only_to_a_lead(client, hosted, lab):
    """Promoting and appointing are two decisions."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_transfer_primary", lab, "member"))

    assert _membership(lab["primary"]).role == Membership.PRIMARY_LEAD
    assert _membership(lab["member"]).role == Membership.MEMBER


def test_transferring_leaves_exactly_one_primary_lead(client, hosted, lab):
    """Both rows change together, so the group never holds two."""
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


def test_a_group_with_people_in_it_is_not_deleted(client, hosted, lab):
    client.force_login(lab["site"])

    client.post(reverse("ui:site_group_delete"), {"pk": lab["group"].pk})

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


def test_a_site_admin_opens_the_bin(client, hosted, lab, django_user_model):
    """The bin page lists each experiment's run count."""
    group, exp = _emptied_group_holding_work(django_user_model)
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 1}, trial_offset=0)
    group.delete()
    client.force_login(lab["site"])

    resp = client.get(reverse("ui:site_trash"))

    assert resp.status_code == 200
    assert [e.run_count for e in resp.context["experiments"]] == [1]


def test_only_a_site_admin_sees_the_bin(client, hosted, lab):
    client.force_login(lab["primary"])

    assert client.get(reverse("ui:site_trash")).status_code == 404


def test_rehoming_puts_it_back_in_a_group(client, hosted, lab,
                                          django_user_model):
    """Rehoming sets the group and clears `trashed_at` together."""
    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    client.force_login(lab["site"])

    client.post(reverse("ui:site_trash_rehome", args=[exp.pk]),
                {"group": lab["group"].pk})
    exp.refresh_from_db()

    assert exp.group_id == lab["group"].pk
    assert exp.trashed_at is None


def test_a_rehomed_experiment_is_its_new_groups(client, hosted, lab,
                                                django_user_model):
    """Rehoming makes the work reachable again."""
    from access.policy import GroupPolicy

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    client.force_login(lab["site"])
    client.post(reverse("ui:site_trash_rehome", args=[exp.pk]),
                {"group": lab["group"].pk})

    request = type("R", (), {"user": lab["primary"]})()
    names = set(GroupPolicy().experiments(request).values_list("data__name", flat=True))

    assert "left behind" in names


def test_discarding_from_the_bin_is_a_second_decision(client, hosted, lab,
                                                      django_user_model):
    """Only `site_trash_delete` can destroy binned work."""
    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    client.force_login(lab["site"])

    client.post(reverse("ui:site_trash_delete", args=[exp.pk]))

    assert not Experiment.objects.filter(pk=exp.pk).exists()
