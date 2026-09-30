"""A person in several groups, and accounts found by email.

*What:* membership is no longer one-per-person. Somebody can be in two groups,
with a different role in each, and every rule that used to ask "their group"
now asks about one named group — so a lead of one group and a member of another
is a lead only where they lead, and nothing from either group leaks into the
other. Alongside that, an email address is how a person is found: adding an
address somebody already has joins that account to the group rather than making
a second one, and a new group cannot exist without a primary lead named by one.

*How:* at the policy for reach, through the panels for adding people, creating
groups and usage, and at the database for the case-insensitive email index —
which has to hold when nobody has been near a view, because the Django admin
edits `User` directly.
"""

import pytest
from django.db import IntegrityError, transaction
from django.urls import reverse

from access.models import Group, Membership
from access.policy import GroupPolicy
from ui.models import Experiment, ExperimentShare, Run
from ui.permissions import DELETE, RUN

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _person(django_user_model, name, *memberships):
    user = django_user_model.objects.create_user(
        username=name, email=f"{name}@example.org", password="pw")
    for group, role in memberships:
        Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


def _experiment(owner, name, group, **kw):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, group=group, **kw)


@pytest.fixture
def two_labs(django_user_model):
    """Two groups, and Mia in both — a lead of `a`, a member of `b`."""
    from django.contrib.auth.models import Permission

    a = Group.objects.create(name="a", user_limit=5)
    b = Group.objects.create(name="b", user_limit=5)
    site = _person(django_user_model, "site")
    site.user_permissions.add(Permission.objects.get(
        content_type__app_label="access", codename="manage_site"))
    return {
        "a": a, "b": b,
        "mia": _person(django_user_model, "mia",
                       (a, Membership.LEAD), (b, Membership.MEMBER)),
        "ann": _person(django_user_model, "ann", (a, Membership.PRIMARY_LEAD)),
        "bob": _person(django_user_model, "bob", (b, Membership.PRIMARY_LEAD)),
        "site": django_user_model.objects.get(pk=site.pk),
    }


def _shared(exp, user, level=ExperimentShare.VIEWER):
    ExperimentShare.objects.create(experiment=exp, user=user, level=level)
    return exp


def _reach(user):
    request = type("R", (), {"user": user})()
    return set(GroupPolicy().experiments(request).values_list("data__name", flat=True))


# ── reach, per group ─────────────────────────────────────────────────────────

def test_somebody_in_two_groups_reaches_each_as_their_role_there(hosted, two_labs):
    """A lead of `a` reaches all of `a`; a member of `b` reaches only what was
    shared with them in `b`. Each group answered on its own, with its own role."""
    _experiment(two_labs["ann"], "a private", two_labs["a"])
    _experiment(two_labs["bob"], "b private", two_labs["b"])
    _shared(_experiment(two_labs["bob"], "b shared", two_labs["b"]), two_labs["mia"])

    assert _reach(two_labs["mia"]) == {"a private", "b shared"}


def test_neither_group_sees_the_other_through_somebody_in_both(hosted, two_labs):
    """Mia being in both is not a bridge: what she owns in `a` is `a`'s, and the
    primary lead of `b` does not reach it by leading somebody who is in `a`."""
    _experiment(two_labs["mia"], "hers in a", two_labs["a"])
    _experiment(two_labs["mia"], "hers in b", two_labs["b"])

    assert _reach(two_labs["bob"]) == {"hers in b"}
    assert _reach(two_labs["ann"]) == {"hers in a"}


def test_leading_one_group_grants_nothing_in_the_other(hosted, two_labs):
    """The rule `may()` applies to a colleague's work asks about the
    experiment's group, not whether the person leads anything at all."""
    exp = _shared(_experiment(two_labs["bob"], "b shared", two_labs["b"]),
                  two_labs["mia"])
    request = type("R", (), {"user": two_labs["mia"]})()

    assert not GroupPolicy().may(request, exp, RUN)
    assert not GroupPolicy().may(request, exp, DELETE)


def test_the_lead_controls_appear_only_where_they_lead(client, hosted, two_labs):
    client.force_login(two_labs["mia"])

    in_a = client.get(reverse("ui:group_detail", args=[two_labs["a"].pk]))
    in_b = client.get(reverse("ui:group_detail", args=[two_labs["b"].pk]))

    assert (in_a.status_code, in_b.status_code) == (200, 200)
    assert in_a.context["is_lead"] and not in_b.context["is_lead"]
    assert reverse("ui:group_add_person", args=[two_labs["b"].pk]) not in in_b.content.decode()


def test_the_group_sidebar_lists_every_group_they_are_in(client, hosted, two_labs):
    client.force_login(two_labs["mia"])

    body = client.get(reverse("ui:group_detail", args=[two_labs["b"].pk])).content.decode()

    for group in ("a", "b"):
        assert reverse("ui:group_detail", args=[two_labs[group].pk]) in body


def test_the_group_tab_lands_on_their_first_group(client, hosted, two_labs):
    client.force_login(two_labs["mia"])

    resp = client.get(reverse("ui:group_index"))

    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:group_detail", args=[two_labs["a"].pk])


def test_removing_somebody_from_one_group_leaves_the_other(client, hosted, two_labs):
    """A removal is of one membership — the account and its other groups stay."""
    client.force_login(two_labs["bob"])
    in_b = two_labs["mia"].memberships.get(group=two_labs["b"])

    client.post(reverse("ui:group_remove_person", args=[two_labs["b"].pk, in_b.pk]))

    assert list(two_labs["mia"].memberships.values_list("group__name", flat=True)) == ["a"]


def test_work_held_in_another_group_does_not_block_a_removal(client, hosted, two_labs):
    """Only what they own *in this group* would be stranded by leaving it."""
    _experiment(two_labs["mia"], "hers in a", two_labs["a"])
    client.force_login(two_labs["bob"])
    in_b = two_labs["mia"].memberships.get(group=two_labs["b"])

    client.post(reverse("ui:group_remove_person", args=[two_labs["b"].pk, in_b.pk]))

    assert not two_labs["mia"].memberships.filter(group=two_labs["b"]).exists()


# ── which group a new experiment is in ───────────────────────────────────────

def test_one_membership_still_decides_the_group(hosted, two_labs):
    exp = Experiment.objects.create(
        name="x", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=two_labs["ann"])

    assert exp.group_id == two_labs["a"].pk


def test_two_memberships_are_not_guessed_between(hosted, two_labs):
    """Filing work under a boundary the owner did not choose is worse than not
    filing it — the caller says which, and the create form will ask."""
    exp = Experiment.objects.create(
        name="x", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=two_labs["mia"])

    assert exp.group_id is None


def test_a_group_the_caller_names_is_kept(hosted, two_labs):
    exp = _experiment(two_labs["mia"], "x", two_labs["b"])

    assert exp.group_id == two_labs["b"].pk


# ── usage ────────────────────────────────────────────────────────────────────

def test_usage_counts_each_run_under_its_own_group_once(client, hosted, two_labs):
    """Somebody in two groups used to have their runs counted under both."""
    exp = _experiment(two_labs["mia"], "hers in a", two_labs["a"])
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 5}, started_by=two_labs["mia"],
                       trial_count=7, trial_offset=0)
    client.force_login(two_labs["site"])

    groups = {g.name: g for g in client.get(reverse("ui:site_usage")).context["groups"]}
    people = {(m.user.username, m.group.name): m
              for m in client.get(reverse("ui:site_usage")).context["people"]}

    assert (groups["a"].runs, groups["a"].trials) == (1, 7)
    assert groups["b"].runs == 0
    assert people[("mia", "a")].runs == 1
    assert people[("mia", "b")].runs == 0


# ── adding somebody by email ─────────────────────────────────────────────────

def test_adding_an_address_somebody_has_joins_that_account(
        client, hosted, two_labs, django_user_model):
    """No second account, and nothing about the existing one changes — in
    particular not its password, which is not the lead's to set."""
    client.force_login(two_labs["ann"])
    before = django_user_model.objects.count()

    client.post(reverse("ui:group_add_person", args=[two_labs["a"].pk]),
                {"email": "BOB@example.org", "password": "new", "role": "member"})

    bob = django_user_model.objects.get(pk=two_labs["bob"].pk)
    assert django_user_model.objects.count() == before
    assert set(bob.memberships.values_list("group__name", flat=True)) == {"a", "b"}
    assert bob.check_password("pw")


def test_adding_an_unknown_address_creates_the_account(
        client, hosted, two_labs, django_user_model):
    """Signed in by the address itself, and told its first password once."""
    client.force_login(two_labs["ann"])

    resp = client.post(reverse("ui:group_add_person", args=[two_labs["a"].pk]),
                       {"email": "New.Person@Example.org", "role": "member"},
                       follow=True)

    user = django_user_model.objects.get(email="new.person@example.org")
    assert user.username == "new.person@example.org"
    assert user.memberships.get().group == two_labs["a"]
    assert "first password" in resp.content.decode()


def test_adding_somebody_already_in_the_group_is_refused(
        client, hosted, two_labs):
    client.force_login(two_labs["ann"])

    resp = client.post(reverse("ui:group_add_person", args=[two_labs["a"].pk]),
                       {"email": "mia@example.org", "role": "member"}, follow=True)

    assert "already in" in resp.content.decode()
    assert two_labs["mia"].memberships.filter(group=two_labs["a"]).count() == 1


def test_an_address_that_is_not_one_creates_nothing(
        client, hosted, two_labs, django_user_model):
    client.force_login(two_labs["ann"])
    before = django_user_model.objects.count()

    client.post(reverse("ui:group_add_person", args=[two_labs["a"].pk]),
                {"email": "not an address", "role": "member"})

    assert django_user_model.objects.count() == before


# ── creating a group ─────────────────────────────────────────────────────────

def test_a_new_group_is_refused_without_a_primary_lead(client, hosted, two_labs):
    """A group nobody can appoint a lead in is a group that cannot grow."""
    client.force_login(two_labs["site"])

    client.post(reverse("ui:site_group_save"), {"name": "c", "user_limit": "3"})

    assert not Group.objects.filter(name="c").exists()


def test_a_new_group_starts_with_its_primary_lead(client, hosted, two_labs,
                                                  django_user_model):
    client.force_login(two_labs["site"])

    client.post(reverse("ui:site_group_save"),
                {"name": "c", "user_limit": "3",
                 "primary_lead_email": "lead.of.c@example.org"})

    group = Group.objects.get(name="c")
    lead = django_user_model.objects.get(email="lead.of.c@example.org")
    assert group.memberships.get().user == lead
    assert group.memberships.get().role == Membership.PRIMARY_LEAD


def test_a_new_groups_primary_lead_may_already_have_an_account(
        client, hosted, two_labs):
    """Which is how a site admin comes to lead a group of their own."""
    client.force_login(two_labs["site"])

    client.post(reverse("ui:site_group_save"),
                {"name": "c", "user_limit": "3",
                 "primary_lead_email": "site@example.org"})

    assert Group.objects.get(name="c").memberships.get().user == two_labs["site"]


def test_a_bad_primary_lead_address_leaves_no_group_behind(client, hosted, two_labs):
    """The two are made together or not at all."""
    client.force_login(two_labs["site"])

    client.post(reverse("ui:site_group_save"),
                {"name": "c", "user_limit": "3", "primary_lead_email": "nope"})

    assert not Group.objects.filter(name="c").exists()


def test_a_new_group_needs_a_seat_for_its_lead(client, hosted, two_labs):
    client.force_login(two_labs["site"])

    client.post(reverse("ui:site_group_save"),
                {"name": "c", "user_limit": "0",
                 "primary_lead_email": "lead.of.c@example.org"})

    assert not Group.objects.filter(name="c").exists()


# ── the database backs it up ─────────────────────────────────────────────────

def test_one_membership_per_person_per_group(hosted, two_labs):
    with pytest.raises(IntegrityError), transaction.atomic():
        Membership.objects.create(user=two_labs["mia"], group=two_labs["a"])


def test_an_address_is_one_account_whatever_its_case(hosted, two_labs,
                                                     django_user_model):
    """The admin's user editor does not go through `find_or_create_by_email`,
    so the index is what stops it making a second Mia."""
    with pytest.raises(IntegrityError), transaction.atomic():
        django_user_model.objects.create_user(username="mia2", email="MIA@example.org")


def test_accounts_without_an_address_do_not_collide(hosted, django_user_model):
    """A stock `createsuperuser` leaves the address blank, and two of them are
    not the same person."""
    django_user_model.objects.create_user(username="one")
    django_user_model.objects.create_user(username="two")

    assert django_user_model.objects.filter(email="").count() == 2
