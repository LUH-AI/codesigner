"""The two management surfaces, and who can open them.

*What:* a group lead gets a Group tab — their group's people, and their
colleagues' experiments in a pane of their own. A site admin gets a Site tab —
groups, usage and running jobs, and **no experiments anywhere**, which is the
claim this file exists to hold.

*How:* through the pages as each role, including the cases that should 404. A
panel nobody may open is not an error page; it is a page that is not there, so
a member asking for the Site tab gets the same answer as somebody asking for a
URL that was never routed.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment, Run

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _person(django_user_model, name, group=None, role=Membership.MEMBER, **kw):
    user = django_user_model.objects.create_user(username=name, password="pw", **kw)
    if group is not None:
        Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


@pytest.fixture
def lab(django_user_model):
    from django.contrib.auth.models import Permission

    group = Group.objects.create(name="lab", user_limit=3)
    other = Group.objects.create(name="other", user_limit=5)
    site = _person(django_user_model, "site")
    site.user_permissions.add(Permission.objects.get(
        content_type__app_label="access", codename="manage_site"))
    return {
        "group": group, "other": other,
        "lead": _person(django_user_model, "vera", group, Membership.LEAD),
        "ana": _person(django_user_model, "ana", group),
        "outsider": _person(django_user_model, "cleo", other),
        "site": django_user_model.objects.get(pk=site.pk),
    }


def _experiment(owner, name, **kw):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, **kw)


# ── who may open what ────────────────────────────────────────────────────────

@pytest.mark.parametrize("route", ["ui:group_people", "ui:group_work"])
def test_only_a_lead_has_a_group_panel(client, hosted, lab, route):
    """A member asking for it gets a 404, not a 403: a panel that is not theirs
    is not a page they were refused, it is a page that is not there."""
    client.force_login(lab["ana"])

    assert client.get(reverse(route)).status_code == 404


@pytest.mark.parametrize("route", ["ui:site_groups", "ui:site_usage", "ui:site_jobs"])
def test_only_a_site_admin_has_a_site_panel(client, hosted, lab, route):
    client.force_login(lab["lead"])

    assert client.get(reverse(route)).status_code == 404


def test_a_lead_opens_their_own_group_and_no_other(client, hosted, lab):
    client.force_login(lab["lead"])

    body = client.get(reverse("ui:group_people")).content.decode()

    assert "lab" in body
    assert "cleo" not in body, "another group's people are not theirs to see"


# ── the seat limit ───────────────────────────────────────────────────────────

def test_a_lead_fills_the_seats_they_were_given(client, hosted, lab,
                                                django_user_model):
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_add_person"),
                {"username": "newbie", "password": "pw", "role": "member"})

    added = django_user_model.objects.get(username="newbie")
    assert added.membership.group == lab["group"]


def test_and_is_refused_past_them_with_the_number(client, hosted, lab):
    """"No" without the count reads as a bug when the limit is somebody else's
    to change."""
    client.force_login(lab["lead"])
    client.post(reverse("ui:group_add_person"),
                {"username": "third", "password": "pw", "role": "member"})

    resp = client.post(reverse("ui:group_add_person"),
                       {"username": "fourth", "password": "pw", "role": "member"},
                       follow=True)

    body = resp.content.decode()
    assert "full" in body and "3" in body
    assert "site admin" in body, "it should say whose limit it is"


def test_a_lead_cannot_raise_their_own_limit(client, hosted, lab):
    """A lead who could would be a lead without one — the save route is the site
    admin's, and it is not theirs to post to."""
    client.force_login(lab["lead"])

    resp = client.post(reverse("ui:site_group_save"),
                       {"pk": lab["group"].pk, "name": "lab", "user_limit": "99"})

    lab["group"].refresh_from_db()
    assert resp.status_code == 404
    assert lab["group"].user_limit == 3


def test_a_site_admin_can(client, hosted, lab):
    client.force_login(lab["site"])

    client.post(reverse("ui:site_group_save"),
                {"pk": lab["group"].pk, "name": "lab", "user_limit": "9",
                 "is_active": "on"})

    lab["group"].refresh_from_db()
    assert lab["group"].user_limit == 9


# ── removing somebody ────────────────────────────────────────────────────────

def test_removing_somebody_who_owns_work_is_refused(client, hosted, lab):
    """Out of the group nobody could reach it — the same silence deleting the
    account would cause, which `access/admin.py` refuses for the same reason."""
    _experiment(lab["ana"], "hers")
    client.force_login(lab["lead"])

    resp = client.post(
        reverse("ui:group_remove_person", args=[lab["ana"].membership.pk]),
        follow=True)

    assert "reassign" in resp.content.decode()
    assert Membership.objects.filter(user=lab["ana"]).exists()


def test_a_lead_cannot_remove_themselves(client, hosted, lab):
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_remove_person",
                        args=[lab["lead"].membership.pk]), follow=True)

    assert Membership.objects.filter(user=lab["lead"]).exists()


def test_a_lead_cannot_reach_into_another_group(client, hosted, lab):
    """The membership is addressed by pk, so this is the route where a wrong
    number would cross the boundary."""
    client.force_login(lab["lead"])

    resp = client.post(reverse("ui:group_remove_person",
                               args=[lab["outsider"].membership.pk]))

    assert resp.status_code == 404
    assert Membership.objects.filter(user=lab["outsider"]).exists()


# ── the lead's two views are different ───────────────────────────────────────

def test_the_group_pane_shows_colleagues_and_not_the_lead(client, hosted, lab):
    # Names without apostrophes: Django escapes them, and asserting against
    # `&#x27;` would be testing the template engine rather than the panel.
    _experiment(lab["ana"], "work belonging to ana")
    _experiment(lab["lead"], "work belonging to vera")
    client.force_login(lab["lead"])

    body = client.get(reverse("ui:group_work")).content.decode()

    assert "work belonging to ana" in body
    assert "work belonging to vera" not in body, \
        "the lead's own stays on the Experiments tab"


# ── what a site admin is not shown ───────────────────────────────────────────

def test_the_site_panels_name_no_experiment(client, hosted, lab):
    """The privacy claim, asserted as an absence — which is the only way to
    assert it, and the reason it is worth a test rather than a comment."""
    exp = _experiment(lab["ana"], "Secret cancer model")
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="running",
                       stopping={"max_trials": 5}, started_by=lab["ana"],
                       job_id="99", trial_offset=0)
    client.force_login(lab["site"])

    for route in ("ui:site_groups", "ui:site_usage", "ui:site_jobs"):
        body = client.get(reverse(route)).content.decode()
        assert "Secret cancer model" not in body, route


def test_a_job_is_identified_without_saying_what_it_is(client, hosted, lab):
    """The experiment's own short code: enough to talk about a job in a support
    conversation, and it says nothing about the work."""
    exp = _experiment(lab["ana"], "Secret cancer model")
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="running",
                       stopping={"max_trials": 5}, started_by=lab["ana"],
                       job_id="99", trial_offset=0)
    client.force_login(lab["site"])

    body = client.get(reverse("ui:site_jobs")).content.decode()

    assert exp.identifier in body
    assert "ana" in body, "who started it is theirs to see"


def test_a_site_admin_can_stop_a_job_they_cannot_see_the_experiment_for(
        client, hosted, lab):
    """Addressed by run, not experiment — there is no experiment visibility for
    `@experiment_view` to check, and granting some so a decorator would fit
    would defeat the separation."""
    exp = _experiment(lab["ana"], "running")
    run = Run.objects.create(experiment=exp, primary_metric="accuracy",
                             status="running", stopping={"max_trials": 5},
                             started_by=lab["ana"], trial_offset=0)
    client.force_login(lab["site"])

    client.post(reverse("ui:site_job_stop", args=[run.pk]), follow=True)

    run.refresh_from_db()
    assert run.status == "error"


def test_nobody_else_can_stop_a_job_that_way(client, hosted, lab):
    exp = _experiment(lab["ana"], "running")
    run = Run.objects.create(experiment=exp, primary_metric="accuracy",
                             status="running", stopping={"max_trials": 5},
                             started_by=lab["ana"], trial_offset=0)
    client.force_login(lab["lead"])

    assert client.post(reverse("ui:site_job_stop", args=[run.pk])).status_code == 404
    run.refresh_from_db()
    assert run.status == "running"


# ── and the tabs only appear for whoever they belong to ─────────────────────

def test_the_rail_offers_each_tab_to_its_own_role(client, hosted, lab):
    for who, expected in [("ana", []), ("vera", ["ui:group_people"]),
                          ("site", ["ui:site_groups"])]:
        client.force_login(lab["site"] if who == "site"
                           else lab["lead"] if who == "vera" else lab["ana"])
        body = client.get(reverse("ui:home")).content.decode()
        for route in ("ui:group_people", "ui:site_groups"):
            present = reverse(route) in body
            assert present == (route in expected), f"{who}: {route}"
