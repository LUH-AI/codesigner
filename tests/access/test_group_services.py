"""The group and user services (access/services/groups.py, users.py) that the
group pages, the management commands and codesigner-manage share: what each
does, and the reason each gives when it cannot proceed.
"""

import pytest

from django.contrib.auth.models import User

from access.models import Account, Group, Membership
from access.services import groups as group_service
from access.services import users as user_service
from access.services.errors import ServiceError
from experiments.models import Experiment

pytestmark = pytest.mark.django_db


def _experiment(owner, group):
    return Experiment.objects.create(
        name="x", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, group=group)


def test_a_group_starts_with_its_primary_lead():
    """What: `create_group` creates the group and, through `LocalOnboarding`,
    an account for its primary lead that has no password until the first
    sign-in. How: a new group with a new username."""
    group, lead, message = group_service.create_group("vision", 3, "vera")

    assert lead.username == "vera" and not lead.has_usable_password()
    assert lead.account.password_pending == Account.FIRST_SIGN_IN
    assert "first time they sign in" in message
    assert group.memberships.get().role == Membership.PRIMARY_LEAD


@pytest.mark.parametrize("name, seats, lead, said", [
    ("", 3, "vera", "needs a name"),
    ("vision", 0, "vera", "at least one seat"),
    ("vision", 3, "", "needs a primary lead"),
    ("vision", 3, "v" * 151, "1 to 150 characters"),
    ("vision", 3, "operator:vera", "only contain letters"),
])
def test_a_group_that_cannot_start_leaves_nothing_behind(name, seats, lead, said):
    """What: `create_group` raises `ServiceError` for a group without a name, a
    seat or a valid lead username, and writes nothing. How: each missing piece
    in turn."""
    with pytest.raises(ServiceError, match=said):
        group_service.create_group(name, seats, lead)
    assert not Group.objects.exists()
    assert not User.objects.exists()


def test_group_names_are_unique_on_create_and_rename():
    """What: `create_group` and `update_group` refuse a name another group
    has. How: a second group named, then renamed, like the first."""
    group_service.create_group("vision", 3, "a@b.de")
    with pytest.raises(ServiceError, match="already a group"):
        group_service.create_group("vision", 3, "c@d.de")
    other, *_ = group_service.create_group("nlp", 3, "c@d.de")
    with pytest.raises(ServiceError, match="already a group"):
        group_service.update_group(other, name="vision", seats=3, active=True)


def test_a_group_with_members_is_not_deleted():
    """What: `delete_group` keeps a group that still has people, and its
    experiments stay in it rather than going to the site bin. How: delete a
    new group whose lead owns one experiment."""
    group, lead, _ = group_service.create_group("vision", 3, "a@b.de")
    experiment = _experiment(lead, group)
    with pytest.raises(ServiceError, match="still has people"):
        group_service.delete_group(group)
    experiment.refresh_from_db()
    assert experiment.group_id == group.pk and experiment.trashed_at is None


def test_adding_people_respects_seats_roles_and_membership():
    """What: `add_person` fills a seat, and refuses a full group, a role the
    adder may not grant and a duplicate. How: a two-seat group."""
    group, *_ = group_service.create_group("vision", 2, "lead@b.de")
    with pytest.raises(ServiceError, match="cannot add somebody with that role"):
        group_service.add_person(group, "ana@b.de", Membership.LEAD,
                                 allowed_roles=[Membership.MEMBER])
    user, created, _ = group_service.add_person(group, "ana@b.de")
    assert created and group.memberships.filter(user=user).exists()
    with pytest.raises(ServiceError, match="is full"):
        group_service.add_person(group, "ben@b.de")
    group.user_limit = 5
    group.save()
    with pytest.raises(ServiceError, match="already in"):
        group_service.add_person(group, "ANA@b.de")


def test_removal_keeps_the_primary_lead_and_owners():
    """What: `remove_person` refuses the primary lead, a person owning work in
    the group and the remover themselves, and removes anybody else. How: a
    group with a lead and two members, one owning an experiment."""
    group, lead_user, *_ = group_service.create_group("vision", 5, "lead@b.de")
    lead = group.memberships.get(user=lead_user)
    owner, *_ = group_service.add_person(group, "owner@b.de")
    other, *_ = group_service.add_person(group, "other@b.de")
    _experiment(owner, group)

    with pytest.raises(ServiceError, match="primary lead"):
        group_service.remove_person(lead)
    with pytest.raises(ServiceError, match="yourself"):
        group_service.remove_person(lead, by=lead)
    with pytest.raises(ServiceError, match="still owns experiments"):
        group_service.remove_person(group.memberships.get(user=owner))
    group_service.remove_person(group.memberships.get(user=other))
    assert not group.memberships.filter(user=other).exists()


def test_the_primary_role_moves_only_to_a_lead():
    """What: `set_role` switches member and lead, and `transfer_primary` hands
    the primary role only to a lead. How: a member, promoted, then given the
    primary role."""
    group, lead_user, *_ = group_service.create_group("vision", 5, "lead@b.de")
    lead = group.memberships.get(user=lead_user)
    ana, *_ = group_service.add_person(group, "ana@b.de")
    member = group.memberships.get(user=ana)

    with pytest.raises(ServiceError, match="only go to a group lead"):
        group_service.transfer_primary(lead, member)
    group_service.set_role(member, Membership.LEAD)
    group_service.transfer_primary(lead, member)

    lead.refresh_from_db()
    member.refresh_from_db()
    assert (lead.role, member.role) == (Membership.LEAD, Membership.PRIMARY_LEAD)


def test_an_owner_is_deactivated_rather_than_deleted(django_user_model):
    """What: `delete_user` refuses an account owning experiments and deletes
    one without; `set_active` blocks and restores sign-in. How: an owner and a
    newcomer."""
    owner = django_user_model.objects.create_user("owner", "owner@b.de", "pw")
    newcomer = django_user_model.objects.create_user("new", "new@b.de", "pw")
    _experiment(owner, None)

    with pytest.raises(ServiceError, match="still owns experiments"):
        user_service.delete_user(owner)
    user_service.set_active(owner, False)
    user_service.delete_user(newcomer)

    assert not django_user_model.objects.get(pk=owner.pk).is_active
    assert not django_user_model.objects.filter(pk=newcomer.pk).exists()


def test_the_custom_model_grant_is_given_and_taken(django_user_model):
    """What: `set_custom_models` grants and withdraws `use_custom_models`.
    How: grant, check, withdraw, check, with the permission cache cleared."""
    user = django_user_model.objects.create_user("ana", "ana@b.de", "pw")
    user_service.set_custom_models(user, True)
    assert django_user_model.objects.get(pk=user.pk).has_perm("access.use_custom_models")
    user_service.set_custom_models(user, False)
    assert not django_user_model.objects.get(pk=user.pk).has_perm("access.use_custom_models")


def test_a_system_account_cannot_be_added_to_a_group(django_user_model):
    """What: `add_person` neither finds nor creates a name outside Django's
    username characters, so a system account such as codesigner-manage's
    `support:dashboard` never joins a group. How: that account made
    beforehand, then added to a group by name."""
    django_user_model.objects.create_user("support:dashboard", is_active=False)
    group, *_ = group_service.create_group("vision", 3, "vera")

    with pytest.raises(ServiceError, match="only contain letters"):
        group_service.add_person(group, "support:dashboard")
    assert group.memberships.count() == 1
