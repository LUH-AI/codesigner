"""Data protection: the deployment's legal pages for everybody, a person's own
data download and deletion, address changes, and the retention sweep."""

import io
import json
import zipfile
from datetime import timedelta

import pytest
from django.core.management import call_command
from django.urls import reverse
from django.utils import timezone

from access.models import AgreementAcceptance, Group, Membership
from access.services import users as user_service
from access.services.errors import ServiceError
from experiments.models import Experiment

pytestmark = pytest.mark.django_db

GOOD = "correct horse battery staple"


def legal_pages():
    """`legal_pages` stands in for a deployment's `LEGAL_PAGES`."""
    return {"privacy": "<p>What is stored, and why.</p>", "imprint": "<p>Who runs this.</p>"}


def terms():
    return "v1", "<p>Be kind to the cluster.</p>"


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    settings.LEGAL_PAGES = f"{__name__}.legal_pages"
    settings.ACCOUNT_AGREEMENT = f"{__name__}.terms"
    return settings


def _person(django_user_model, email, group=None, role=Membership.MEMBER):
    user = django_user_model.objects.create_user(email, email=email, password=GOOD)
    AgreementAcceptance.objects.create(user=user, version="v1")
    if group is not None:
        Membership.objects.create(user=user, group=group, role=role)
    return user


def _experiment(owner, group=None, name="study"):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, group=group)


# Legal pages

def test_the_legal_pages_are_open_to_everybody(client, hosted):
    """What: the imprint and the terms answer without signing in; the terms
    page carries the privacy policy after the terms, and the privacy address
    leads to that section. The sign-in page links the imprint and the combined
    page. How: GETs while signed out."""
    assert "Who runs this." in client.get(reverse("ui:imprint")).content.decode()
    page = client.get(reverse("ui:terms")).content.decode()
    assert page.index("Be kind to the cluster.") < page.index("What is stored")
    resp = client.get(reverse("ui:privacy"))
    assert resp.status_code == 302 and resp["Location"] == reverse("ui:terms") + "#privacy"
    login = client.get(reverse("ui:login")).content.decode()
    assert reverse("ui:imprint") in login and reverse("ui:terms") in login


def test_without_terms_the_privacy_policy_has_its_own_page(client, hosted):
    """What: with only `LEGAL_PAGES` set, the privacy policy shows on its own
    page. How: unset `ACCOUNT_AGREEMENT`, open the privacy page."""
    hosted.ACCOUNT_AGREEMENT = ""
    page = client.get(reverse("ui:privacy"))
    assert page.status_code == 200 and "What is stored" in page.content.decode()


def test_the_privacy_policy_is_readable_before_the_terms_are_accepted(
        client, hosted, django_user_model):
    """What: somebody held at the terms can still open the privacy policy, and
    the terms page links it. How: an account with no acceptance."""
    user = django_user_model.objects.create_user("ana@lab.de", email="ana@lab.de",
                                                 password=GOOD)
    client.force_login(user)

    assert client.get(reverse("ui:privacy"), follow=True).status_code == 200
    gate = client.get(reverse("ui:agreement")).content.decode()
    assert reverse("ui:privacy") in gate


def test_without_the_hooks_there_are_no_legal_pages(client, settings):
    """What: an install without `LEGAL_PAGES` or `ACCOUNT_AGREEMENT` has no
    such pages and no links. How: the pages and the sign-in page with the
    hooks unset."""
    settings.REQUIRE_LOGIN = True
    for name in ("ui:imprint", "ui:privacy", "ui:terms"):
        assert client.get(reverse(name)).status_code == 404
    assert "legal-links" not in client.get(reverse("ui:login")).content.decode()


# Download

def test_a_person_downloads_everything_held_about_them(client, hosted, django_user_model):
    """What: "Download my data" gives a zip with the account, memberships,
    accepted terms, assistant chats and an `.ihpo` per experiment owned. How:
    one member with an experiment and a chat."""
    from assistant.models import Conversation

    group = Group.objects.create(name="lab", user_limit=3)
    ana = _person(django_user_model, "ana@lab.de", group)
    exp = _experiment(ana, group)
    chat = Conversation.objects.create(owner=ana)
    chat.messages.create(role="user", content="Which model suits wine?")
    client.force_login(ana)

    response = client.post(reverse("ui:account_data"))
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    data = json.loads(bundle.read("account.json"))

    assert response["Content-Type"] == "application/zip"
    assert data["account"]["email"] == "ana@lab.de"
    assert data["memberships"][0]["group"] == "lab"
    assert data["accepted_terms"][0]["version"] == "v1"
    assert data["assistant_conversations"][0]["messages"][0]["text"] == "Which model suits wine?"
    assert f"experiments/{exp.identifier}.ihpo" in bundle.namelist()


# Erasure

def test_deleting_your_account_hands_your_experiments_to_the_group(
        client, hosted, django_user_model):
    """What: a member's own deletion passes their experiments to the group's
    primary lead, removes the account with its chats, actions and sessions,
    and signs them out. How: the page with a wrong password, then the right
    one."""
    from assistant.models import AgentAction, Conversation

    group = Group.objects.create(name="lab", user_limit=3)
    vera = _person(django_user_model, "vera@lab.de", group, Membership.PRIMARY_LEAD)
    ana = _person(django_user_model, "ana@lab.de", group)
    exp = _experiment(ana, group)
    Conversation.objects.create(owner=ana)
    AgentAction.objects.create(user=ana, tool="run", outcome="done")
    client.force_login(ana)
    url = reverse("ui:account_delete")

    client.post(url, {"password": "wrong", "email": "ana@lab.de"})
    assert django_user_model.objects.filter(pk=ana.pk).exists()

    done = client.post(url, {"password": GOOD, "email": "ANA@lab.de"})

    exp.refresh_from_db()
    assert done.status_code == 302 and "_auth_user_id" not in client.session
    assert not django_user_model.objects.filter(pk=ana.pk).exists()
    assert exp.owner == vera and exp.deleted_at is None
    assert not Conversation.objects.exists() and not AgentAction.objects.exists()


def test_a_primary_lead_hands_the_role_on_or_is_told_why_not(django_user_model):
    """What: a primary lead's deletion promotes the longest-serving lead; with
    only plain members left it is refused and nothing changes. How: two
    groups, one with a lead and one without."""
    led = Group.objects.create(name="led", user_limit=5)
    unled = Group.objects.create(name="unled", user_limit=5)
    vera = _person(django_user_model, "vera@lab.de", led, Membership.PRIMARY_LEAD)
    Membership.objects.create(user=vera, group=unled, role=Membership.PRIMARY_LEAD)
    ben = _person(django_user_model, "ben@lab.de", led, Membership.LEAD)
    _person(django_user_model, "cleo@lab.de", unled)

    with pytest.raises(ServiceError, match="unled"):
        user_service.erase_user(vera)
    assert Membership.objects.get(user=vera, group=led).role == Membership.PRIMARY_LEAD

    Membership.objects.filter(user=vera, group=unled).delete()
    user_service.erase_user(vera)

    assert Membership.objects.get(user=ben, group=led).role == Membership.PRIMARY_LEAD


def test_an_only_member_s_experiments_go_to_the_bin(django_user_model):
    """What: with nobody else in the group, deleting the account bins the
    experiments, which then wait in the site bin. How: one person alone in a
    group."""
    from experiments.services import site as site_service

    group = Group.objects.create(name="solo", user_limit=1)
    ana = _person(django_user_model, "ana@lab.de", group, Membership.PRIMARY_LEAD)
    exp = _experiment(ana, group)

    user_service.erase_user(ana)

    exp.refresh_from_db()
    assert exp.deleted_at is not None and exp.owner is None
    assert exp.pk in site_service.trash().values_list("pk", flat=True)


def test_an_operator_cannot_delete_a_primary_lead_out_of_a_group(django_user_model):
    """What: `delete_user` refuses a primary lead of a group with other people
    in it. How: a primary lead with one member beside them."""
    group = Group.objects.create(name="lab", user_limit=3)
    vera = _person(django_user_model, "vera@lab.de", group, Membership.PRIMARY_LEAD)
    _person(django_user_model, "ana@lab.de", group)

    with pytest.raises(ServiceError, match="primary lead"):
        user_service.delete_user(vera)


# Rectification

def test_an_operator_changes_somebody_s_address(django_user_model):
    """What: `users set-email` changes the address and the username behind it,
    and refuses one another account uses or that is no address. How: two
    accounts and three changes."""
    ana = _person(django_user_model, "ana@lab.de")
    _person(django_user_model, "ben@lab.de")
    out = io.StringIO()

    call_command("users", "set-email", "ana@lab.de", "Ana.New@Lab.de", json=True, stdout=out)

    ana.refresh_from_db()
    assert (ana.email, ana.username) == ("ana.new@lab.de", "ana.new@lab.de")
    with pytest.raises(ServiceError, match="Another account"):
        user_service.set_email(ana, "ben@lab.de")
    with pytest.raises(ServiceError, match="not an email"):
        user_service.set_email(ana, "nobody")


# Retention

def test_the_sweep_deletes_only_what_is_past_its_retention(settings, django_user_model):
    """What: `privacy sweep` deletes expired sessions, failed sign-ins past
    their days and chats untouched past theirs, keeps the rest, and only
    counts with `--dry-run`. How: one old and one new row of each kind."""
    from axes.models import AccessFailureLog
    from django.contrib.sessions.backends.db import SessionStore
    from django.contrib.sessions.models import Session

    from assistant.models import Conversation

    now = timezone.now()
    for days in (0, -1):
        store = SessionStore()
        store.create()
        Session.objects.filter(session_key=store.session_key).update(
            expire_date=now + timedelta(days=days or 1))
    for days in (1, 40):
        row = AccessFailureLog.objects.create(username="ana@lab.de", ip_address="10.0.0.1",
                                              user_agent="x", path_info="/", locked_out=False)
        AccessFailureLog.objects.filter(pk=row.pk).update(attempt_time=now - timedelta(days=days))
    ana = _person(django_user_model, "ana@lab.de")
    for days in (1, 120):
        chat = Conversation.objects.create(owner=ana)
        Conversation.objects.filter(pk=chat.pk).update(updated_at=now - timedelta(days=days))

    def sweep(*flags):
        out = io.StringIO()
        call_command("privacy", "sweep", *flags, json=True, stdout=out)
        return json.loads(out.getvalue())

    counted = sweep("--dry-run")
    assert (counted["sessions"], counted["failed_sign_ins"], counted["conversations"]) == (1, 1, 1)
    assert Session.objects.count() == 2

    sweep()
    assert Session.objects.count() == 1
    assert AccessFailureLog.objects.count() == 1
    assert Conversation.objects.count() == 1
