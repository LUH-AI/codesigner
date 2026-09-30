"""Groups do not see each other.

*What:* an instance hosted for several research groups exists so that the groups
are separate. Everything in `access/policy.py` narrows towards that, and this is
where the narrowing is checked — at the queryset, where a miss is a 404 and an
experiment you may not see is indistinguishable from one that is not there.

*How:* two groups with a lead and members each, plus a site admin in neither,
and one experiment at each sharing level: not shared, a viewer, a contributor. Cross-group access is asserted to
404 rather than 403, because a 403 would confirm the experiment exists.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment, ExperimentShare
from ui.permissions import DELETE, EDIT, EXPORT, RUN, SHARE, VIEW

VIEWER, CONTRIBUTOR = ExperimentShare.VIEWER, ExperimentShare.CONTRIBUTOR

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


def _experiment(owner, name, *, shares=None):
    exp = Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner)
    for user, level in (shares or {}).items():
        ExperimentShare.objects.create(experiment=exp, user=user, level=level)
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
    w["ana_viewer"] = _experiment(w["ana"], "ana viewer", shares={w["ben"]: VIEWER})
    w["ana_contrib"] = _experiment(w["ana"], "ana contrib",
                                   shares={w["ben"]: CONTRIBUTOR})
    w["cleo_private"] = _experiment(w["cleo"], "cleo private")
    return w


def _visible(client, user):
    """The names this user's sidebar would list."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": user})()
    return set(GroupPolicy().experiments(request).values_list("data__name", flat=True))


def _detail(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk]))


# ── the boundary ─────────────────────────────────────────────────────────────

def test_a_member_sees_their_own_and_what_was_shared_with_them(hosted, world):
    assert _visible(None, world["ben"]) == {"ana viewer", "ana contrib"}
    assert _visible(None, world["ana"]) == {"ana private", "ana viewer", "ana contrib"}


def test_another_group_sees_none_of_it(hosted, world):
    """The property the whole feature exists for."""
    assert _visible(None, world["cleo"]) == {"cleo private"}
    assert _visible(None, world["lead_n"]) == {"cleo private"}


def test_reaching_across_groups_is_a_404_not_a_403(client, hosted, world):
    """A 403 would confirm it exists. An experiment you may not see has to be
    indistinguishable from one that is not there, or the URL space reports who
    has what."""
    client.force_login(world["cleo"])

    assert _detail(client, world["ana_viewer"]).status_code == 404
    assert _detail(client, world["ana_private"]).status_code == 404


def test_a_group_lead_reaches_nothing_in_another_group(client, hosted, world):
    """Being a lead is a fact inside one group, not a rank across the site."""
    client.force_login(world["lead_n"])

    assert _detail(client, world["ana_viewer"]).status_code == 404


# ── the three sharing levels ─────────────────────────────────────────────────

def test_unshared_is_the_owners_alone(hosted, world):
    assert "ana private" not in _visible(None, world["ben"])


def test_a_grant_cannot_cross_the_group_boundary(hosted, world):
    """Sharing is how work moves within a group, not out of one."""
    from django.core.exceptions import ValidationError

    grant = ExperimentShare(experiment=world["ana_private"], user=world["cleo"])

    with pytest.raises(ValidationError):
        grant.full_clean()


def test_the_view_refuses_it_too(client, hosted, world):
    """Not found, rather than refused: the form names a colleague, and somebody
    outside the group is not one."""
    client.force_login(world["ana"])

    resp = client.post(reverse("ui:experiment_share", args=[world["ana_private"].pk]),
                       {"user": world["cleo"].pk, "level": VIEWER})

    assert resp.status_code == 404
    assert not world["ana_private"].shares.exists()


def test_a_grant_that_got_across_anyway_reaches_nothing(hosted, world):
    """The admin and a data migration are routes that skip `clean()`. The
    policy asks about membership itself, so a stray row grants nothing."""
    ExperimentShare.objects.create(experiment=world["ana_private"],
                                   user=world["cleo"], level=CONTRIBUTOR)

    assert "ana private" not in _visible(None, world["cleo"])


def test_a_viewer_may_look_and_not_touch(client, hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["ben"]})()
    policy = GroupPolicy()

    assert policy.may(request, world["ana_viewer"], VIEW)
    assert policy.may(request, world["ana_viewer"], EXPORT)
    assert not policy.may(request, world["ana_viewer"], RUN)
    assert not policy.may(request, world["ana_viewer"], EDIT)
    assert not policy.may(request, world["ana_viewer"], DELETE)


def test_a_contributor_may_do_everything_but_share_it(client, hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["ben"]})()
    policy = GroupPolicy()

    for action in (VIEW, EXPORT, RUN, EDIT, DELETE):
        assert policy.may(request, world["ana_contrib"], action), action
    assert not policy.may(request, world["ana_contrib"], SHARE)


def test_leaving_the_group_takes_the_grants_in_it(hosted, world):
    """Out of the group, a grant would be access across the boundary."""
    Membership.objects.filter(user=world["ben"]).delete()

    assert not ExperimentShare.objects.filter(user=world["ben"]).exists()


def test_a_grant_does_not_follow_the_work_to_another_group(hosted, world):
    """Rehoming moves the experiment and not its colleagues: Ben is not in the
    group it went to, so what he was given is no longer his to reach."""
    Experiment.objects.filter(pk=world["ana_viewer"].pk).update(group=world["nlp"])

    assert "ana viewer" not in _visible(None, world["ben"])


# ── the lead's two views ─────────────────────────────────────────────────────

def test_a_lead_reaches_everything_in_their_group(hosted, world):
    assert _visible(None, world["lead_v"]) == {"ana private", "ana viewer",
                                               "ana contrib"}


def test_but_their_everyday_list_stays_their_own(hosted, world):
    """Being a lead should not quietly turn the main list into everyone's."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()
    listed = set(GroupPolicy().for_listing(request).values_list("data__name", flat=True))

    assert listed == set(), "a colleague's work is on the group page, not here"


def test_the_group_panel_is_the_other_half(hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()
    panel = set(GroupPolicy().group_experiments(request).values_list("data__name", flat=True))

    assert panel == {"ana private", "ana viewer", "ana contrib"}


def test_a_lead_may_run_their_groups_work(hosted, world):
    """Which is the point of the role: stopping a run that is going wrong
    should not need its owner to be awake."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()

    assert GroupPolicy().may(request, world["ana_private"], RUN)
    assert GroupPolicy().may(request, world["ana_private"], EDIT)


def test_but_not_delete_it(hosted, world):
    """Deleting is a decision about the work, and the work is its people's —
    the owner and whoever they made a contributor."""
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["lead_v"]})()

    assert not GroupPolicy().may(request, world["ana_private"], DELETE)
    assert not GroupPolicy().may(request, world["ana_private"], SHARE)


def test_unless_they_were_made_a_contributor(hosted, world):
    from access.policy import GroupPolicy

    ExperimentShare.objects.create(experiment=world["ana_private"],
                                   user=world["lead_v"], level=CONTRIBUTOR)
    request = type("R", (), {"user": world["lead_v"]})()

    assert GroupPolicy().may(request, world["ana_private"], DELETE)


def test_a_member_may_not(hosted, world):
    from access.policy import GroupPolicy

    request = type("R", (), {"user": world["ben"]})()

    assert not GroupPolicy().may(request, world["ana_viewer"], RUN)


# ── the site admin ───────────────────────────────────────────────────────────

def test_a_site_admin_in_no_group_sees_no_experiments(hosted, world):
    """Not because of the role — because there is nothing of theirs to see.

    `manage_site` says what they may administer, and their membership says what
    they may read. The site admin in `world` has none, so the second answer is
    empty and the first never enters into it.
    """
    assert _visible(None, world["site"]) == set()


def test_a_site_admin_cannot_reach_another_groups_by_url(client, hosted, world):
    client.force_login(world["site"])

    assert _detail(client, world["ana_viewer"]).status_code == 404


def test_a_site_admin_with_a_group_sees_that_group(hosted, world, django_user_model):
    """`create_site_admin` gives them a group of one by default, and that group
    is theirs to work in.

    Administering the instance and belonging to a group are separate facts, so
    holding `manage_site` neither grants reach nor withdraws it: this reads
    exactly like any other lead of vision-lab, and nlp-group stays invisible.
    """
    Membership.objects.create(user=world["site"], group=world["vision"],
                              role=Membership.PRIMARY_LEAD)
    site = django_user_model.objects.get(pk=world["site"].pk)

    assert _visible(None, site) == {"ana private", "ana viewer", "ana contrib"}
    assert "cleo private" not in _visible(None, site)


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
