"""The two management surfaces, and who can open them.

*What:* everybody with an account gets a Group tab — a page per group they are
in, with its people and its experiments as they may reach them. A lead's page
adds their colleagues' experiments and the controls for the group's people. A site admin gets a Site tab —
groups, usage, running jobs and the bin, and **nobody else's experiments
anywhere on it**, which is the claim this file exists to hold. What a site admin
may read is their own membership's business and is checked in `test_groups.py`;
these panels show none of it either way.

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

def test_a_member_opens_their_group_without_the_lead_controls(client, hosted, lab):
    """Who is in the group is everybody's to see; changing it is the leads'."""
    client.force_login(lab["ana"])

    resp = client.get(reverse("ui:group_detail", args=[lab["group"].pk]))
    body = resp.content.decode()

    assert resp.status_code == 200
    assert "vera" in body, "the member list is read-only, not hidden"
    assert reverse("ui:group_add_person", args=[lab["group"].pk]) not in body
    assert "/remove/" not in body


def test_somebody_outside_it_gets_a_404(client, hosted, lab):
    """Not a 403: a page that is not theirs is not a page they were refused, it
    is a page that is not there."""
    client.force_login(lab["outsider"])

    assert client.get(reverse("ui:group_detail", args=[lab["group"].pk])
                      ).status_code == 404


def test_a_member_cannot_post_to_the_lead_controls(client, hosted, lab):
    client.force_login(lab["ana"])

    resp = client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                       {"email": "x@example.org", "role": "member"})

    assert resp.status_code == 404


@pytest.mark.parametrize("route", ["ui:site_groups", "ui:site_usage", "ui:site_jobs"])
def test_only_a_site_admin_has_a_site_panel(client, hosted, lab, route):
    client.force_login(lab["lead"])

    assert client.get(reverse(route)).status_code == 404


def test_a_lead_opens_their_own_group_and_no_other(client, hosted, lab):
    client.force_login(lab["lead"])

    body = client.get(reverse("ui:group_detail", args=[lab["group"].pk])).content.decode()

    assert "lab" in body
    assert "cleo" not in body, "another group's people are not theirs to see"
    assert client.get(reverse("ui:group_detail", args=[lab["other"].pk])
                      ).status_code == 404


# ── the seat limit ───────────────────────────────────────────────────────────

def test_a_lead_fills_the_seats_they_were_given(client, hosted, lab,
                                                django_user_model):
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                {"email": "newbie@example.org", "password": "pw", "role": "member"})

    added = django_user_model.objects.get(email="newbie@example.org")
    assert added.memberships.get().group == lab["group"]


def test_and_is_refused_past_them_with_the_number(client, hosted, lab):
    """"No" without the count reads as a bug when the limit is somebody else's
    to change."""
    client.force_login(lab["lead"])
    add = reverse("ui:group_add_person", args=[lab["group"].pk])
    client.post(add, {"email": "third@example.org", "role": "member"})

    resp = client.post(add, {"email": "fourth@example.org", "role": "member"},
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
        reverse("ui:group_remove_person", args=[lab["group"].pk, lab["ana"].memberships.get().pk]),
        follow=True)

    assert "reassign" in resp.content.decode()
    assert Membership.objects.filter(user=lab["ana"]).exists()


def test_a_lead_cannot_remove_themselves(client, hosted, lab):
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_remove_person",
                        args=[lab["group"].pk, lab["lead"].memberships.get().pk]),
                follow=True)

    assert Membership.objects.filter(user=lab["lead"]).exists()


def test_a_lead_cannot_reach_into_another_group(client, hosted, lab):
    """The membership is addressed by pk, so this is the route where a wrong
    number would cross the boundary."""
    client.force_login(lab["lead"])

    resp = client.post(reverse("ui:group_remove_person",
                               args=[lab["group"].pk,
                                     lab["outsider"].memberships.get().pk]))

    assert resp.status_code == 404
    assert Membership.objects.filter(user=lab["outsider"]).exists()


# ── what the group page lists ────────────────────────────────────────────────

def _page(client, lab, who):
    client.force_login(lab[who])
    return client.get(reverse("ui:group_detail", args=[lab["group"].pk])).context


def test_a_lead_sees_everybody_elses_apart_from_their_own(client, hosted, lab):
    _experiment(lab["ana"], "hers")
    _experiment(lab["lead"], "vera's")

    page = _page(client, lab, "lead")

    assert [e.name for e in page["others"]] == ["hers"]
    assert [row["experiment"].name for row in page["mine"]] == ["vera's"]


def test_a_member_sees_their_own_and_what_was_shared_with_them(client, hosted, lab):
    from ui.models import ExperimentShare

    _experiment(lab["lead"], "vera's, unshared")
    shared = _experiment(lab["lead"], "vera's, shared")
    ExperimentShare.objects.create(experiment=shared, user=lab["ana"],
                                   level=ExperimentShare.CONTRIBUTOR)

    page = _page(client, lab, "ana")

    assert [(e.name, e.level) for e in page["shared"]] == [("vera's, shared", "contributor")]
    assert page["others"] == [], "a member does not reach unshared work"


def test_the_sharing_controls_name_every_colleague(client, hosted, lab):
    """And only colleagues: a grant can only name somebody in the group."""
    _experiment(lab["ana"], "hers")

    row = _page(client, lab, "ana")["mine"][0]

    assert {c["user"].username for c in row["colleagues"]} == {"vera"}
    assert all(c["level"] == "none" for c in row["colleagues"])


def test_a_share_set_from_the_group_page_goes_back_to_it(client, hosted, lab):
    exp = _experiment(lab["ana"], "hers")
    client.force_login(lab["ana"])
    here = reverse("ui:group_detail", args=[lab["group"].pk])

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": lab["lead"].pk, "level": "viewer",
                        "next": f"{here}#exp-{exp.pk}"})

    assert resp["Location"] == f"{here}#exp-{exp.pk}"
    assert exp.shares.get().user == lab["lead"]


def test_the_experiment_page_points_to_where_sharing_is(client, hosted, lab):
    exp = _experiment(lab["ana"], "hers")
    client.force_login(lab["ana"])

    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert reverse("ui:group_detail", args=[lab["group"].pk]) in body
    assert 'name="shared"' not in body, "the old checkbox is gone"


def test_somebody_in_no_group_is_told_so(client, hosted, lab):
    client.force_login(lab["site"])

    resp = client.get(reverse("ui:group_index"))

    assert resp.status_code == 200
    assert "not in a group" in resp.content.decode()


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
    """The Group tab is everybody's; the Site tab only a site admin's."""
    for who, expected in [("ana", ["ui:group_index"]), ("vera", ["ui:group_index"]),
                          ("site", ["ui:group_index", "ui:site_groups"])]:
        client.force_login(lab["site"] if who == "site"
                           else lab["lead"] if who == "vera" else lab["ana"])
        body = client.get(reverse("ui:home")).content.decode()
        for route in ("ui:group_index", "ui:site_groups"):
            present = reverse(route) in body
            assert present == (route in expected), f"{who}: {route}"
