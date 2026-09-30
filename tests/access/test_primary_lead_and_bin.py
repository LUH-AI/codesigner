"""The primary lead, and what a deleted group leaves behind.

*What:* two rules laid on top of groups. Exactly one lead per group is its
primary one, and only they may make or unmake the others; the role moves by
transfer and is never simply dropped, so a group can end up with neither two of
them nor none. And deleting a group does not destroy what it held — the work
goes to a bin a site admin settles one experiment at a time, which is what
makes deleting a group a decision that can be taken quickly.

*How:* through the panels as each role, including the cases that must 404, and
at the model for the invariants a form is not the only route to. The database
constraint and `Group.delete()` both have to hold when nobody has been near a
view, because the admin and a data migration are routes too.
"""

import pytest
from django.db import IntegrityError, transaction
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment

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
    """One group with a primary lead, an ordinary lead and a member."""
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
    """A membership route: addressed by the group and the row within it."""
    return reverse(route, args=[lab["group"].pk, _membership(lab[who]).pk])


# ── who may appoint a lead ───────────────────────────────────────────────────

def test_an_ordinary_lead_has_no_role_controls(client, hosted, lab):
    """The one thing the primary role carries. A lead asking for it gets a 404
    rather than a refusal, like every other panel that is not theirs."""
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
    """Standing down is a transfer, which names a successor. Doing it here
    would leave the group with nobody able to appoint one."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_set_role", lab, "primary"),
                {"role": Membership.MEMBER})

    assert _membership(lab["primary"]).role == Membership.PRIMARY_LEAD


def test_the_role_is_not_settable_to_primary(client, hosted, lab):
    """It moves by transfer and only by transfer, or two rows could claim it."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_set_role", lab, "lead"),
                {"role": Membership.PRIMARY_LEAD})

    assert _membership(lab["lead"]).role == Membership.LEAD


def test_an_ordinary_lead_may_only_add_members(client, hosted, lab,
                                               django_user_model):
    """Otherwise the one restriction the primary role carries is one form post
    away from meaningless."""
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                {"email": "newbie@example.org", "role": Membership.LEAD})

    assert not django_user_model.objects.filter(email="newbie@example.org").exists()


# ── transferring it ──────────────────────────────────────────────────────────

def test_the_role_transfers_only_to_a_lead(client, hosted, lab):
    """Handing it to a member would promote and appoint in a single step, which
    is two decisions wearing one button."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_transfer_primary", lab, "member"))

    assert _membership(lab["primary"]).role == Membership.PRIMARY_LEAD
    assert _membership(lab["member"]).role == Membership.MEMBER


def test_transferring_leaves_exactly_one_primary_lead(client, hosted, lab):
    """Both rows move together: the group has one before and one after, and
    never two in between."""
    client.force_login(lab["primary"])

    client.post(_in_lab("ui:group_transfer_primary", lab, "lead"))

    assert _membership(lab["lead"]).role == Membership.PRIMARY_LEAD
    assert _membership(lab["primary"]).role == Membership.LEAD
    assert Membership.objects.filter(group=lab["group"],
                                     role=Membership.PRIMARY_LEAD).count() == 1


def test_a_group_cannot_hold_two_primary_leads(hosted, lab):
    """Held in the database, not only in the view that transfers it: the admin
    and a data migration are routes to this table too."""
    membership = _membership(lab["lead"])
    membership.role = Membership.PRIMARY_LEAD

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            membership.save(update_fields=["role"])


def test_the_primary_lead_cannot_be_removed_from_the_group(client, hosted, lab):
    """Their group would be left unable to appoint a lead. The role has to be
    transferred first, which is a decision rather than a side effect."""
    client.force_login(lab["primary"])
    # Another lead trying it, since the primary lead is already barred from
    # removing themselves.
    client.post(_in_lab("ui:group_remove_person", lab, "primary"))

    assert Membership.objects.filter(user=lab["primary"]).exists()


# ── a new experiment knows its group ─────────────────────────────────────────

def test_a_new_experiment_lands_in_its_owners_group(hosted, lab):
    """Recorded on the experiment rather than derived from the owner, so that
    deleting the account or moving them does not move the work."""
    exp = _experiment(lab["member"], "theirs")

    assert exp.group_id == lab["group"].pk


def test_an_owner_deleted_leaves_the_work_in_its_group(hosted, lab):
    """The case the stored group exists for: `owner` goes null and there is
    nothing left to derive a group from."""
    exp = _experiment(lab["member"], "theirs")
    lab["member"].delete()
    exp.refresh_from_db()

    assert exp.owner_id is None
    assert exp.group_id == lab["group"].pk


# ── the bin ──────────────────────────────────────────────────────────────────

def _emptied_group_holding_work(django_user_model):
    """A group with no people left but an experiment still in it.

    Which is the only state a group can be deleted from — `Membership.group` is
    PROTECT — and exactly the state the bin exists for: the account went, the
    results did not.
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
    """Emptying a group of its accounts is a separate decision, made person by
    person. One button should not take it as a side effect."""
    client.force_login(lab["site"])

    client.post(reverse("ui:site_group_delete"), {"pk": lab["group"].pk})

    assert Group.objects.filter(pk=lab["group"].pk).exists()


def test_binned_work_is_visible_to_nobody(hosted, lab, django_user_model):
    """Out of every group, so every rule about who may see it would otherwise
    have to special-case it. The bin is its own question."""
    from access.policy import GroupPolicy

    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()

    for user in (lab["primary"], lab["member"], lab["site"]):
        request = type("R", (), {"user": user})()
        names = GroupPolicy().experiments(request).values_list("data__name", flat=True)
        assert "left behind" not in set(names)


def test_only_a_site_admin_sees_the_bin(client, hosted, lab):
    client.force_login(lab["primary"])

    assert client.get(reverse("ui:site_trash")).status_code == 404


def test_rehoming_puts_it_back_in_a_group(client, hosted, lab,
                                          django_user_model):
    """And it stops being binned in the same move — the two are one fact said
    twice, so nothing should be able to set one without the other."""
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
    """What rehoming is *for*: somebody can reach it again."""
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
    """The only route by which deleting a group eventually destroys anything."""
    group, exp = _emptied_group_holding_work(django_user_model)
    group.delete()
    client.force_login(lab["site"])

    client.post(reverse("ui:site_trash_delete", args=[exp.pk]))

    assert not Experiment.objects.filter(pk=exp.pk).exists()
