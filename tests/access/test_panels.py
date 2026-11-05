"""Every account gets a Group tab, and a lead's group page adds colleagues'
experiments and people controls. Panels outside a user's role return 404. Site
administration has no pages at all; its services (`manage.py jobs`, `usage`)
identify work without naming it.
"""

import io
import json

import pytest
from django.core.management import call_command
from django.urls import reverse

from access.models import Group, Membership
from experiments.models import Experiment, Run

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
    group = Group.objects.create(name="lab", user_limit=3)
    other = Group.objects.create(name="other", user_limit=5)
    return {
        "group": group, "other": other,
        "lead": _person(django_user_model, "vera", group, Membership.LEAD),
        "ana": _person(django_user_model, "ana", group),
        "outsider": _person(django_user_model, "cleo", other),
        "nobody": _person(django_user_model, "nils"),
    }


def _printed(*args):
    out = io.StringIO()
    call_command(*args, json=True, stdout=out)
    return out.getvalue()


def _experiment(owner, name, **kw):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, **kw)


def test_a_member_opens_their_group_without_the_lead_controls(client, hosted, lab):
    """`group_detail` shows a member the people list without lead controls."""
    client.force_login(lab["ana"])

    resp = client.get(reverse("ui:group_detail", args=[lab["group"].pk]))
    body = resp.content.decode()

    assert resp.status_code == 200
    assert "vera" in body, "the member list is read-only, not hidden"
    assert reverse("ui:group_add_person", args=[lab["group"].pk]) not in body
    assert "/remove/" not in body


def test_somebody_outside_it_gets_a_404(client, hosted, lab):
    """Another group's page looks unrouted to an outsider."""
    client.force_login(lab["outsider"])

    assert client.get(reverse("ui:group_detail", args=[lab["group"].pk])
                      ).status_code == 404


def test_a_member_cannot_post_to_the_lead_controls(client, hosted, lab):
    client.force_login(lab["ana"])

    resp = client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                       {"identifier": "x", "role": "member"})

    assert resp.status_code == 404


@pytest.mark.parametrize("path", ["/site/", "/site/usage/", "/site/jobs/", "/site/trash/",
                                  "/admin/"])
def test_the_site_has_no_administration_pages(client, hosted, django_user_model, path):
    """What: neither the old site pages nor Django's admin exist, even for a
    superuser. How: each old address, signed in as a superuser."""
    client.force_login(django_user_model.objects.create_superuser("root", "r@x.de", "pw"))

    assert client.get(path).status_code == 404


def test_a_lead_opens_their_own_group_and_no_other(client, hosted, lab):
    client.force_login(lab["lead"])

    body = client.get(reverse("ui:group_detail", args=[lab["group"].pk])).content.decode()

    assert "lab" in body
    assert "cleo" not in body, "another group's people are not theirs to see"
    assert client.get(reverse("ui:group_detail", args=[lab["other"].pk])
                      ).status_code == 404


def test_a_lead_fills_the_seats_they_were_given(client, hosted, lab,
                                                django_user_model):
    client.force_login(lab["lead"])

    client.post(reverse("ui:group_add_person", args=[lab["group"].pk]),
                {"identifier": "newbie@lab.de", "role": "member"})

    added = django_user_model.objects.get(email="newbie@lab.de")
    assert added.memberships.get().group == lab["group"]


def test_and_is_refused_past_them_with_the_number(client, hosted, lab):
    """The refusal states the count and whose limit applies."""
    client.force_login(lab["lead"])
    add = reverse("ui:group_add_person", args=[lab["group"].pk])
    client.post(add, {"identifier": "third", "role": "member"})

    resp = client.post(add, {"identifier": "fourth", "role": "member"},
                       follow=True)

    body = resp.content.decode()
    assert "full" in body and "3" in body
    assert "site admin" in body, "it should say whose limit it is"


def test_removing_somebody_who_owns_work_is_refused(client, hosted, lab):
    """Removing an owner would strand the work, as deleting the account would."""
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
    """The route takes a membership pk from any group."""
    client.force_login(lab["lead"])

    resp = client.post(reverse("ui:group_remove_person",
                               args=[lab["group"].pk,
                                     lab["outsider"].memberships.get().pk]))

    assert resp.status_code == 404
    assert Membership.objects.filter(user=lab["outsider"]).exists()


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
    from experiments.models import ExperimentShare

    _experiment(lab["lead"], "vera's, unshared")
    shared = _experiment(lab["lead"], "vera's, shared")
    ExperimentShare.objects.create(experiment=shared, user=lab["ana"],
                                   level=ExperimentShare.CONTRIBUTOR)

    page = _page(client, lab, "ana")

    assert [(e.name, e.level) for e in page["shared"]] == [("vera's, shared", "contributor")]
    assert page["others"] == [], "a member does not reach unshared work"


def test_the_sharing_controls_name_every_colleague(client, hosted, lab):
    """A grant can only name group members."""
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


def test_only_somebody_elses_experiment_says_whose_it_is(client, hosted, lab):
    """What: the owner's page carries no line about sharing, and a lead
    looking at a member's experiment is told whose it is. How: Ana's
    experiment opened by Ana, then by her lead."""
    exp = _experiment(lab["ana"], "hers")

    client.force_login(lab["ana"])
    own = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    client.force_login(lab["lead"])
    led = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'class="caption ownership"' not in own
    assert reverse("ui:group_detail", args=[lab["group"].pk]) + "#" not in own
    assert 'class="caption ownership"' in led and "in a group you lead" in led


def test_somebody_in_no_group_is_told_so(client, hosted, lab):
    client.force_login(lab["nobody"])

    resp = client.get(reverse("ui:group_index"))

    assert resp.status_code == 200
    assert "not in a group" in resp.content.decode()


def test_the_site_reports_name_no_experiment(hosted, lab):
    """What: the group, usage and jobs reports never name an experiment, only
    its identifier and who started it. How: a running job on an experiment
    with a telling name, through each command's JSON."""
    exp = _experiment(lab["ana"], "Secret cancer model")
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="running",
                       stopping={"max_trials": 5}, started_by=lab["ana"],
                       job_id="99", trial_offset=0)

    for args in (("groups", "list"), ("usage", "groups"), ("usage", "people"),
                 ("jobs", "list")):
        assert "Secret cancer model" not in _printed(*args), args
    job = json.loads(_printed("jobs", "list"))[0]
    assert job["identifier"] == exp.identifier and job["who"] == "ana"


def test_a_stopped_job_keeps_count_of_its_trials(hosted, lab):
    """What: `jobs stop` ends a run and counts its trials past `trial_offset`,
    as a page stop does. How: a running run with three stored trials and an
    offset of one."""
    exp = _experiment(lab["ana"], "running")
    exp.data.result = {"data": [{}, {}, {}]}
    exp.data.save(update_fields=["result"])
    run = Run.objects.create(experiment=exp, primary_metric="accuracy",
                             status="running", stopping={"max_trials": 5},
                             started_by=lab["ana"], trial_offset=1)

    _printed("jobs", "stop", str(run.pk))

    run.refresh_from_db()
    assert run.status == "error" and run.trial_count == 2


def test_the_rail_offers_the_group_tab_and_no_site_tab(client, hosted, lab):
    """What: every account's rail offers the Group tab and none offers a Site
    tab. How: a member's and a lead's home page."""
    for who in ("ana", "lead"):
        client.force_login(lab[who])
        body = client.get(reverse("ui:home")).content.decode()
        assert reverse("ui:group_index") in body, who
        assert "/site/" not in body, who
