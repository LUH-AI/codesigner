"""How an account gets its password and keeps the site's terms: the first
sign-in of a `LocalOnboarding` account, one-time `SetupLink`s, the agreement
`ACCOUNT_AGREEMENT` supplies, `.AccountGateMiddleware`, `.SiteBackend`, the
sweep of unused accounts, and the password rules.
"""

from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone

from access.models import Account, AgreementAcceptance, Group, Membership, SetupLink
from access.services import groups as group_service
from access.services import setup_links
from tests.access.test_cli import run

pytestmark = pytest.mark.django_db

GOOD = "correct horse battery staple"


def terms():
    """`terms` stands in for a deployment's `ACCOUNT_AGREEMENT`."""
    return "v1", "<p>Be kind to the cluster.</p>"


def newer_terms():
    return "v2", "<p>Be kinder to the cluster.</p>"


def pages():
    """`pages` stands in for a deployment's `LEGAL_PAGES`."""
    return {"privacy": "<p>What is stored.</p>", "imprint": "<p>Who runs this.</p>"}


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


@pytest.fixture
def lab():
    return Group.objects.create(name="lab", user_limit=9)


def _added(lab, name="newbie@lab.de"):
    user, created, _ = group_service.add_person(lab, name)
    assert created
    return user


def _set(client, url, password=GOOD, **extra):
    return client.post(url, {"new_password1": password, "new_password2": password, **extra})


def _sign_in(client, address="newbie@lab.de"):
    resp = client.post(reverse("ui:login"), {"username": address, "password": GOOD})
    assert resp.status_code == 302


def test_a_new_account_chooses_its_password_at_first_sign_in(client, hosted, lab):
    """What: an address a lead added signs in without a password, chooses
    one, is sent to the sign-in page and signs in with it; afterwards the
    password is required. How: sign in as the new account, set a password,
    sign in with it, sign out, sign in by address alone."""
    user = _added(lab)

    resp = client.post(reverse("ui:login"), {"username": "newbie@lab.de", "password": ""})
    assert resp.status_code == 302 and resp["Location"] == reverse("ui:first_password")
    resp = _set(client, reverse("ui:first_password"))
    assert resp.status_code == 302 and resp["Location"] == reverse("ui:login")
    assert "_auth_user_id" not in client.session
    _sign_in(client)
    assert client.get(reverse("ui:home")).status_code == 200

    user.refresh_from_db()
    assert user.check_password(GOOD) and user.account.password_pending == ""
    client.logout()
    resp = client.post(reverse("ui:login"), {"username": "newbie@lab.de", "password": ""})
    assert resp.status_code == 200 and "do not match" in resp.content.decode()


def test_the_first_password_follows_the_password_rules(client, hosted, lab):
    """What: the first password needs 15 characters and is not a common one.
    How: a 14-character password, then a common one, then a good one."""
    _added(lab)
    client.post(reverse("ui:login"), {"username": "newbie@lab.de", "password": ""})

    for weak in ("abcdefghijklmn", "1q2w3e4r5t6y7u8i9o0p"):
        resp = _set(client, reverse("ui:first_password"), weak)
        assert resp.status_code == 200 and resp.context["form"].errors
    assert _set(client, reverse("ui:first_password")).status_code == 302


def test_the_first_password_page_needs_the_sign_in_first(client, hosted, lab):
    """What: the first-password page without an address named on the sign-in
    page sends the visitor back there. How: open it directly."""
    _added(lab)

    resp = client.get(reverse("ui:first_password"))

    assert resp.status_code == 302 and resp["Location"] == reverse("ui:login")


def test_people_sign_in_by_username_or_address(client, hosted, django_user_model):
    """What: `.SiteBackend` accepts the username or the email address, in any
    case. How: one account signing in both ways."""
    django_user_model.objects.create_user("ana", "ana@lab.de", GOOD)

    for name in ("ANA", "Ana@Lab.de"):
        assert client.login(username=name, password=GOOD)
        client.logout()


@pytest.mark.parametrize("flag", ["is_staff", "is_superuser"])
def test_operator_accounts_cannot_sign_in(client, hosted, django_user_model, flag):
    """What: staff and superuser accounts, which belong to the operators'
    tools, cannot sign in to the site even with the right password. How: the
    sign-in form with correct credentials."""
    django_user_model.objects.create_user("root", password=GOOD, **{flag: True})

    resp = client.post(reverse("ui:login"), {"username": "root", "password": GOOD})

    assert resp.status_code == 200 and "_auth_user_id" not in client.session


def test_a_setup_link_sets_the_password_once(client, hosted, lab):
    """What: a setup link survives being opened, sets the password and leads
    to the sign-in page with a confirmation, then is dead. How: open the link
    twice, post a password, follow the redirect, open the link again."""
    user = _added(lab)
    url = setup_links.issue(user, hours=24)
    assert user.account.password_pending == Account.SETUP_LINK

    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 200
    resp = _set(client, url)
    assert resp.status_code == 302 and resp["Location"] == reverse("ui:login")
    assert "_auth_user_id" not in client.session
    assert "Your account is ready" in client.get(resp["Location"]).content.decode()

    user.refresh_from_db()
    assert user.check_password(GOOD) and user.account.password_pending == ""
    client.logout()
    assert client.get(url).status_code == 410


def test_a_setup_link_expires(client, hosted, lab):
    """What: a link past its time, or one replaced by a newer link, opens the
    expired page. How: issue two links and age the second."""
    user = _added(lab)
    first = setup_links.issue(user, hours=24)
    second = setup_links.issue(user, hours=24)
    SetupLink.objects.filter(used_at__isnull=True).update(
        expires_at=timezone.now() - timedelta(seconds=1))

    assert client.get(first).status_code == 410
    assert client.get(second).status_code == 410


def test_a_setup_link_asks_for_the_agreement(client, hosted, lab):
    """What: with `ACCOUNT_AGREEMENT` set, the setup page links the terms and
    sets nothing until they are accepted. How: post without, then with,
    acceptance."""
    hosted.ACCOUNT_AGREEMENT = "tests.access.test_onboarding.terms"
    user = _added(lab)
    url = setup_links.issue(user, hours=24)

    page = client.get(url).content.decode()
    assert "By clicking the box below you agree" in page
    assert page.count(f'href="{reverse("ui:terms")}"') == 1
    assert _set(client, url).status_code == 200
    assert not AgreementAcceptance.objects.exists()
    assert _set(client, url, accept="on").status_code == 302
    assert AgreementAcceptance.objects.get().version == "v1"


def test_with_a_privacy_policy_both_boxes_are_required(client, hosted, lab):
    """What: with `LEGAL_PAGES` also set, the setup page has a second box
    confirming the privacy policy was read, and sets nothing until both are
    ticked; the re-acceptance page needs both too. How: post with only the
    terms box, then with both; then the agreement page the same way."""
    hosted.ACCOUNT_AGREEMENT = "tests.access.test_onboarding.terms"
    hosted.LEGAL_PAGES = "tests.access.test_onboarding.pages"
    user = _added(lab)
    url = setup_links.issue(user, hours=24)

    assert 'name="privacy_read"' in client.get(url).content.decode()
    assert _set(client, url, accept="on").status_code == 200
    assert not AgreementAcceptance.objects.exists()
    assert _set(client, url, accept="on", privacy_read="on").status_code == 302
    assert AgreementAcceptance.objects.get().version == "v1"

    hosted.ACCOUNT_AGREEMENT = "tests.access.test_onboarding.newer_terms"
    _sign_in(client)
    client.post(reverse("ui:agreement"), {"accept": "1"})
    assert AgreementAcceptance.objects.count() == 1
    client.post(reverse("ui:agreement"), {"accept": "1", "privacy_read": "1"})
    assert AgreementAcceptance.objects.count() == 2


def test_the_gate_holds_a_person_at_a_new_agreement(client, hosted, lab,
                                                   django_user_model):
    """What: a signed-in person who has not accepted the current agreement is
    sent to it: a page by redirect, an htmx request by `HX-Redirect`, any other
    request by 403. How: each kind of request, then acceptance."""
    hosted.ACCOUNT_AGREEMENT = "tests.access.test_onboarding.terms"
    ana = django_user_model.objects.create_user("ana", password=GOOD)
    client.force_login(ana)

    resp = client.get(reverse("ui:home"))
    assert resp.status_code == 302 and resp["Location"].startswith(reverse("ui:agreement"))
    resp = client.get(reverse("ui:home"), headers={"HX-Request": "true"})
    assert resp.status_code == 204 and resp["HX-Redirect"] == reverse("ui:agreement")
    assert client.post(reverse("ui:home")).status_code == 403

    client.post(reverse("ui:agreement"), {"accept": "1"})
    assert client.get(reverse("ui:home")).status_code == 200


def test_a_required_new_password_comes_before_anything_else(client, hosted,
                                                            django_user_model):
    """What: after `users require-new-password`, the person is sent to change
    their password, and a change frees them. How: require it, open the home
    page, change the password, open it again."""
    ana = django_user_model.objects.create_user("ana", password=GOOD)
    client.force_login(ana)
    assert run("users", "require-new-password", "ana") == {"required": 1}

    resp = client.get(reverse("ui:home"))
    assert resp.status_code == 302 and resp["Location"].startswith(reverse("ui:password_change"))
    client.post(reverse("ui:password_change"), {
        "old_password": GOOD, "new_password1": GOOD + " again",
        "new_password2": GOOD + " again"})

    assert client.get(reverse("ui:home")).status_code == 200


def test_require_new_password_all_leaves_out_pending_and_operator_accounts(
        lab, django_user_model):
    """What: `--all` marks every active person with a password, and nobody
    still waiting for one or with an operator flag. How: one of each."""
    django_user_model.objects.create_user("ana", password=GOOD)
    django_user_model.objects.create_user("root", password=GOOD, is_superuser=True)
    _added(lab)

    assert run("users", "require-new-password", "--all") == {"required": 1}
    assert list(Account.objects.filter(must_change_password=True)
                .values_list("user__username", flat=True)) == ["ana"]


def test_resetting_a_password_returns_to_first_sign_in(client, hosted, django_user_model):
    """What: `users reset-password` under `LocalOnboarding` makes the old
    password useless and lets the next sign-in choose one. How: reset, then
    sign in by username alone."""
    django_user_model.objects.create_user("ana", password=GOOD)

    assert "next time they sign in" in run("users", "reset-password", "ana")

    resp = client.post(reverse("ui:login"), {"username": "ana", "password": ""})
    assert resp["Location"] == reverse("ui:first_password")


def test_the_sweep_removes_accounts_whose_links_lapsed(lab, django_user_model):
    """What: `setup_links sweep` deletes an account whose links expired
    unused, with its membership, and keeps one with a live link or one that
    signed in. How: three added people in those states."""
    lapsed, waiting, joined = (_added(lab, f"{name}@lab.de")
                               for name in ("lapsed", "waiting", "joined"))
    for user in (lapsed, waiting, joined):
        setup_links.issue(user, hours=24)
    SetupLink.objects.filter(user=lapsed).update(expires_at=timezone.now())
    joined.set_password(GOOD)
    joined.last_login = timezone.now()
    joined.save()

    assert run("setup_links", "sweep") == {"removed": ["lapsed@lab.de"]}
    assert set(Membership.objects.values_list("user__email", flat=True)) == {
        "waiting@lab.de", "joined@lab.de"}


def test_setup_links_issue_prints_a_full_link(lab):
    """What: `setup_links issue` prints the link under `--base-url`. How: one
    link for a new account."""
    _added(lab)

    issued = run("setup_links", "issue", "newbie@lab.de", "--base-url", "https://example.org/")

    assert issued["link"].startswith("https://example.org/accounts/setup/")


def test_local_onboarding_offers_no_reset_page(client, hosted):
    """What: `LocalOnboarding` has no mail to reset by, so the forgotten-password
    page does not exist and the sign-in page does not link it. How: both
    pages."""
    assert client.get(reverse("ui:password_forgot")).status_code == 404
    assert reverse("ui:password_forgot") not in client.get(reverse("ui:login")).content.decode()


def test_wrong_passwords_lock_the_username_out(client, hosted, settings, django_user_model):
    """What: django-axes locks a username from one address after
    `AXES_FAILURE_LIMIT` wrong passwords, so even the right one then fails.
    How: three wrong attempts with a limit of three, then the right one."""
    settings.AXES_ENABLED = True
    settings.AXES_FAILURE_LIMIT = 3
    django_user_model.objects.create_user("ana", password=GOOD)

    for _ in range(3):
        client.post(reverse("ui:login"), {"username": "ana", "password": "wrong"})
    resp = client.post(reverse("ui:login"), {"username": "ana", "password": GOOD})

    assert resp.status_code == 429 and "_auth_user_id" not in client.session


def test_every_spelling_of_an_address_shares_one_lockout(client, hosted, settings,
                                                         django_user_model):
    """What: failures count against the account, so changing the letter case
    of the address or adding spaces gives no fresh attempts.
    How: three wrong attempts under three spellings with a limit of three, then
    the right password under a fourth."""
    settings.AXES_ENABLED = True
    settings.AXES_FAILURE_LIMIT = 3
    django_user_model.objects.create_user("ana@lab.de", email="ana@lab.de", password=GOOD)

    for typed in ("ana@lab.de", "Ana@Lab.de", " ANA@LAB.DE "):
        client.post(reverse("ui:login"), {"username": typed, "password": "wrong"})
    resp = client.post(reverse("ui:login"), {"username": "aNa@lab.DE", "password": GOOD})

    assert resp.status_code == 429 and "_auth_user_id" not in client.session


def test_a_forged_forwarded_for_header_gives_no_fresh_attempts(client, hosted, settings,
                                                               django_user_model):
    """What: the lockout counts the socket address, which a client-sent
    `X-Forwarded-For` leaves alone.
    How: three wrong attempts, each claiming a different address in the header,
    with a limit of three, then the right password."""
    settings.AXES_ENABLED = True
    settings.AXES_FAILURE_LIMIT = 3
    django_user_model.objects.create_user("ana@lab.de", email="ana@lab.de", password=GOOD)

    for n in range(3):
        client.post(reverse("ui:login"), {"username": "ana@lab.de", "password": "wrong"},
                    HTTP_X_FORWARDED_FOR=f"198.51.100.{n}, 203.0.113.7")
    resp = client.post(reverse("ui:login"), {"username": "ana@lab.de", "password": GOOD},
                       HTTP_X_FORWARDED_FOR="198.51.100.9, 203.0.113.7")

    assert resp.status_code == 429 and "_auth_user_id" not in client.session


def test_the_shipped_password_rules():
    """What: outside the tests, passwords hash with Argon2 and need 15
    characters. How: the settings module itself, which the suite's fixtures
    leave alone."""
    import config.settings as shipped

    assert shipped.PASSWORD_HASHERS[0].endswith("Argon2PasswordHasher")
    lengths = [v.get("OPTIONS", {}).get("min_length") for v in shipped.AUTH_PASSWORD_VALIDATORS
               if v["NAME"].endswith("MinimumLengthValidator")]
    assert lengths == [15]


def test_a_setup_link_points_at_the_site_whoever_issues_it(settings, django_user_model):
    """What: `setup_links.issue` builds the link from `SITE_URLCONF`, so a
    project that mounts the site's pages under a prefix, as codesigner-manage
    does under /support/, still mails a link to the site's own page. How:
    a URL table with the site under /support/ as `ROOT_URLCONF`."""
    import sys
    import types

    from django.urls import include, path

    from access.services import setup_links

    mounted = types.ModuleType("mounted_urls")
    mounted.urlpatterns = [path("support/", include("config.urls"))]
    sys.modules["mounted_urls"] = mounted
    settings.ROOT_URLCONF = "mounted_urls"

    link = setup_links.issue(django_user_model.objects.create_user("ana"), 24)

    assert link.startswith("/accounts/setup/")


def test_a_display_name_is_chosen_with_the_password(client, hosted, lab):
    """What: the setup page takes an optional display name of at least three
    characters with the password, its placeholder the email address, and the group page then shows it beside
    the email address. How: a two-letter name, then a good one."""
    user = _added(lab)
    Membership.objects.filter(user=user).update(role=Membership.PRIMARY_LEAD)
    url = setup_links.issue(user, hours=24)
    assert 'placeholder="newbie@lab.de"' in client.get(url).content.decode()

    resp = _set(client, url, display_name="Al")
    assert resp.status_code == 200 and "display_name" in resp.context["form"].errors
    assert _set(client, url, display_name="  Ada   Lovelace ").status_code == 302
    _sign_in(client)

    user.account.refresh_from_db()
    assert user.account.display_name == "Ada Lovelace"
    page = client.get(reverse("ui:group_detail", args=[lab.pk])).content.decode()
    assert "Ada Lovelace (newbie@lab.de)" in page


def test_the_display_name_changes_on_the_account_page(client, hosted, django_user_model):
    """What: the account page changes the display name, refuses one under
    three characters, and an empty one shows the email address again. How:
    three posts to the page."""
    from access.models import shown_name

    user = django_user_model.objects.create_user("ana@lab.de", "ana@lab.de", GOOD)
    client.force_login(user)
    url = reverse("ui:account")

    client.post(url, {"display_name": "Ana"})
    assert shown_name(django_user_model.objects.get(pk=user.pk)) == "Ana"
    resp = client.post(url, {"display_name": "A"})
    assert resp.status_code == 200 and resp.context["name_form"].errors
    client.post(url, {"display_name": ""})
    assert shown_name(django_user_model.objects.get(pk=user.pk)) == "ana@lab.de"
