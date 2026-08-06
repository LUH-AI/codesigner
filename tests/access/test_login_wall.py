"""The login wall, and the two things that have to stay outside it.

`REQUIRE_LOGIN` is the whole switch. Off, the wall must be invisible — that is
the primary case, and a local install regressing into a login prompt would be
the worst outcome of this package existing at all. On, everything is gated
*except* the healthcheck and the language switcher, and the tests below pin why
each exception is there rather than just that it is.

The wall is default-closed: Django's `LoginRequiredMiddleware` gates a view
unless it is marked exempt, so a route added later is protected by having been
forgotten about. `test_the_url_space_has_no_unintended_exemptions` is what keeps
the exempt list from quietly growing.
"""

import pytest
from django.urls import get_resolver, reverse

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    """An instance with accounts."""
    settings.REQUIRE_LOGIN = True


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="pw")


# ── off: nothing changes ─────────────────────────────────────────────────────

def test_without_the_setting_no_page_asks_for_a_login(client, settings):
    """The primary case. Most installs are one person on their own machine."""
    settings.REQUIRE_LOGIN = False

    for name in ("ui:home", "ui:new_experiment", "ui:import_experiment",
                 "ui:appearance", "ui:default_experiment_settings"):
        assert client.get(reverse(name)).status_code == 200, name


def test_the_wall_is_installed_even_when_it_is_off():
    """Inert, not absent: the hosted and unhosted configurations are the same
    configuration, so turning the switch on cannot miss a step."""
    from django.conf import settings as django_settings

    assert "access.middleware.LoginWallMiddleware" in django_settings.MIDDLEWARE


def test_the_switch_is_read_per_request(client, settings):
    """Not captured at startup — an operator's `.env` and a test both change it
    after the middleware was constructed."""
    settings.REQUIRE_LOGIN = True
    assert client.get(reverse("ui:home")).status_code == 302

    settings.REQUIRE_LOGIN = False
    assert client.get(reverse("ui:home")).status_code == 200


# ── on: the wall ─────────────────────────────────────────────────────────────

def test_a_stranger_is_sent_to_the_login_page(client, hosted):
    resp = client.get(reverse("ui:home"))

    assert resp.status_code == 302
    assert resp["Location"].startswith(reverse("access:login"))


def test_the_page_they_wanted_is_carried_across_the_login(client, hosted):
    """A shared link to an experiment should still land on that experiment."""
    target = reverse("ui:appearance")
    resp = client.get(target)

    assert f"next={target}" in resp["Location"]


def test_signing_in_returns_them_to_it(client, hosted, user):
    target = reverse("ui:appearance")

    resp = client.post(reverse("access:login"),
                       {"username": "ana", "password": "pw", "next": target})

    assert resp.status_code == 302
    assert resp["Location"] == target


def test_a_signed_in_user_sees_the_app(client, hosted, user):
    client.force_login(user)
    assert client.get(reverse("ui:home")).status_code == 200


def test_a_wrong_password_says_so_rather_than_signing_in(client, hosted, user):
    resp = client.post(reverse("access:login"),
                       {"username": "ana", "password": "not-it"})

    assert resp.status_code == 200
    assert not resp.wsgi_request.user.is_authenticated


def test_signing_out_needs_a_post(client, hosted, user):
    """A GET logout can be triggered by anything that makes the browser fetch a
    URL — an <img> tag on another site is enough."""
    client.force_login(user)
    assert client.get(reverse("access:logout")).status_code == 405


def test_signing_out_ends_the_session(client, hosted, user):
    client.force_login(user)

    client.post(reverse("access:logout"))

    assert client.get(reverse("ui:home")).status_code == 302


# ── on: the two exemptions ───────────────────────────────────────────────────

def test_the_healthcheck_answers_without_a_session(client, hosted):
    """The container runtime has no session. A healthcheck that 302s to a login
    page reports a healthy instance as down, and the orchestrator restarts it
    forever."""
    resp = client.get(reverse("ui:healthz"))

    assert resp.status_code == 200
    assert resp.content == b"ok"


def test_the_language_can_be_switched_before_signing_in(client, hosted):
    """The login page carries the switcher, so this has to work while gated —
    otherwise the one page a stranger can reach is the one page they may not be
    able to read."""
    resp = client.post(reverse("set_language"),
                       {"language": "de", "next": reverse("access:login")})

    assert resp.status_code == 302
    assert client.cookies["django_language"].value == "de"


def test_the_login_page_renders_in_the_chosen_language(client, hosted):
    client.post(reverse("set_language"),
                {"language": "de", "next": reverse("access:login")})

    html = client.get(reverse("access:login")).content.decode()

    assert "Anmelden" in html


def test_the_url_space_has_no_unintended_exemptions(hosted):
    """The wall is default-closed, so an exemption is always deliberate. This
    lists the deliberate ones; anything else appearing here is a hole."""
    exempt = set()
    for pattern in get_resolver().url_patterns:
        for route in getattr(pattern, "url_patterns", [pattern]):
            callback = getattr(route, "callback", None)
            if callback is not None and getattr(callback, "login_required", True) is False:
                exempt.add(route.name)

    assert exempt == {"healthz", "set_language", "login"}
