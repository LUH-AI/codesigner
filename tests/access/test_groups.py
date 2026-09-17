"""Groups do not see each other.

*What:* an instance hosted for several research groups exists so that the groups
are separate. Everything in `access/policy.py` narrows towards that, and this is
where the narrowing is checked — at the queryset, where a miss is a 404 and an
experiment you may not see is indistinguishable from one that is not there.

*How:* two groups with a lead and members each, plus a site admin in neither,
and an experiment at all three sharing levels. Cross-group access is asserted to
404 rather than 403, because a 403 would confirm the experiment exists.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment
from ui.permissions import DELETE, EXPORT, RUN, VIEW

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _user(django_user_model, name, group=None, role=Membership.MEMBER):
    user = django_user_model.objects.create_user(username=name, password="pw")
    if group is not None:
        Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


def _experiment(owner, name, *, shared=False, shared_with=()):
    exp = Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, shared=shared)
    exp.shared_with.set(shared_with)
    return exp


@pytest.fixture
def world(django_user_model):
    """Two groups, and a site admin belonging to neither."""
    from django.contrib.auth.models import Permission

    vision = Group.objects.create(name="vision-lab", user_limit=5)
    nlp = Group.objects.create(name="nlp-group", user_limit=3)

    w = {
        "vision": vision, "nlp": nlp,
        "lead_v": _user(django_user_model, "lead_v", vision, Membership.LEAD),
        "ana": _user(django_user_model, "ana", vision),
        "ben": _user(django_user_model, "ben", vision),
        "lead_n": _user(django_user_model, "lead_n", nlp, Membership.LEAD),
        "cleo": _user(django_user_model, "cleo", nlp),
        "site": django_user_model.objects.create_user("site", password="pw"),
    }
    w["site"].user_permissions.add(Permission.objects.get(
        content_type__app_label="access", codename="manage_site"))
    w["site"] = django_user_model.objects.get(pk=w["site"].pk)

    w["ana_private"] = _experiment(w["ana"], "ana private")
    w["ana_group"] = _experiment(w["ana"], "ana group", shared=True)
    w["ana_named"] = _experiment(w["ana"], "ana named", shared_with=[w["ben"]])
    w["cleo_group"] = _experiment(w["cleo"], "cleo group", shared=True)
    return w


def _visible(client, user):
    """The names this user's sidebar would list."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": user})()
    return set(GroupPolicy().experiments(request).values_list("name", flat=True))


def _detail(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk]))


# ── the boundary ─────────────────────────────────────────────────────────────

def test_a_member_sees_their_own_and_what_their_group_shared(hosted, world):
    assert _visible(None, world["ben"]) == {"ana group", "ana named"}
    assert _visible(None, world["ana"]) == {"ana private", "ana group", "ana named"}


def test_another_group_sees_none_of_it(hosted, world):
    """The property the whole feature exists for."""
    assert _visible(None, world["cleo"]) == {"cleo group"}
    assert _visible(None, world["lead_n"]) == {"cleo group"}


def test_reaching_across_groups_is_a_404_not_a_403(client, hosted, world):
    """A 403 would confirm it exists. An experiment you may not see has to be
    indistinguishable from one that is not there, or the URL space reports who
    has what."""
    client.force_login(world["cleo"])

    assert _detail(client, world["ana_group"]).status_code == 404
    assert _detail(client, world["ana_private"]).status_code == 404


def test_a_group_lead_reaches_nothing_in_another_group(client, hosted, world):
    """Being a lead is a fact inside one group, not a rank across the site."""
    client.force_login(world["lead_n"])

    assert _detail(client, world["ana_group"]).status_code == 404


# ── the three sharing levels ─────────────────────────────────────────────────

def test_unshared_is_the_owners_alone(hosted, world):
    assert "ana private" not in _visible(None, world["ben"])


def test_sharing_with_a_name_crosses_the_group_boundary(hosted, world,
                                                        django_user_model):
    """Deliberately. An invitation the owner made by name is an act of judgement
    rather than a property of a group, so it is not bounded by one."""
    world["ana_named"].shared_with.add(world["cleo"])

    assert "ana named" in _visible(None, world["cleo"])


def test_a_named_share_grants_looking_and_not_touching(client, hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["ben"]})()
    policy = GroupPolicy()

    assert policy.may(request, world["ana_named"], VIEW)
    assert policy.may(request, world["ana_named"], EXPORT)
    assert not policy.may(request, world["ana_named"], RUN)
    assert not policy.may(request, world["ana_named"], DELETE)


# ── the lead's two views ─────────────────────────────────────────────────────

def test_a_lead_reaches_everything_in_their_group(hosted, world):
    assert _visible(None, world["lead_v"]) == {"ana private", "ana group",
                                               "ana named"}


def test_but_their_everyday_list_stays_their_own(hosted, world):
    """Being a lead should not quietly turn the main list into everyone's."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()
    listed = set(GroupPolicy().for_listing(request).values_list("name", flat=True))

    assert "ana private" not in listed, "a colleague's unshared work is not theirs to list"
    assert "ana group" in listed, "what the group shared is still group-visible"


def test_the_group_panel_is_the_other_half(hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()
    panel = set(GroupPolicy().group_experiments(request).values_list("name", flat=True))

    assert panel == {"ana private", "ana group", "ana named"}


def test_a_lead_may_act_on_their_groups_work(hosted, world):
    """Which is the point of the role: stopping a run that is going wrong
    should not need its owner to be awake."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()

    assert GroupPolicy().may(request, world["ana_private"], RUN)
    assert GroupPolicy().may(request, world["ana_private"], DELETE)


def test_a_member_may_not(hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["ben"]})()

    assert not GroupPolicy().may(request, world["ana_group"], RUN)


# ── the site admin ───────────────────────────────────────────────────────────

def test_a_site_admin_sees_no_experiments_at_all(hosted, world):
    """The role is defined by not seeing them."""
    assert _visible(None, world["site"]) == set()


def test_a_site_admin_cannot_reach_one_by_url(client, hosted, world):
    client.force_login(world["site"])

    assert _detail(client, world["ana_group"]).status_code == 404


def test_even_with_a_group_of_their_own(hosted, world, django_user_model):
    """`create_site_admin` gives them one by default, and it must not become a
    way to see that group's work."""
    Membership.objects.create(user=world["site"], group=world["vision"],
                              role=Membership.LEAD)
    site = django_user_model.objects.get(pk=world["site"].pk)

    assert _visible(None, site) == set()


# ── what ownerless means now ─────────────────────────────────────────────────

def test_an_ownerless_experiment_is_nobodys(hosted, world):
    """It used to be everyone's, which was right while there was one flat pool
    of accounts and wrong the moment there are groups — every member of every
    group would see it. The migration gave the existing ones an owner."""
    _experiment(None, "orphan")

    assert "orphan" not in _visible(None, world["ana"])
    assert "orphan" not in _visible(None, world["lead_v"])


# ── and none of this applies without accounts ────────────────────────────────

def test_without_accounts_everything_is_everyones(settings, world):
    settings.REQUIRE_LOGIN = False

    assert len(_visible(None, world["cleo"])) == 4
