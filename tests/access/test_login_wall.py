"""`REQUIRE_LOGIN` switches the wall, and a local install with the switch off
never sees a login prompt. `LoginWallMiddleware` exempts the login page, the
healthcheck and the language switcher.
"""

import pytest
from django.urls import get_resolver, reverse

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="pw")


def test_without_the_setting_no_page_asks_for_a_login(client, settings):
    """Most installs are one person on one machine."""
    settings.REQUIRE_LOGIN = False

    for name in ("ui:home", "ui:new_experiment", "ui:import_experiment",
                 "ui:appearance", "ui:default_experiment_settings"):
        assert client.get(reverse(name)).status_code == 200, name


def test_the_wall_is_installed_even_when_it_is_off():
    """One configuration serves both modes, so turning the switch on cannot miss
    a step."""
    from django.conf import settings as django_settings

    assert "access.middleware.LoginWallMiddleware" in django_settings.MIDDLEWARE


def test_the_switch_is_read_per_request(client, settings):
    """Tests change the setting after the middleware is built."""
    settings.REQUIRE_LOGIN = True
    assert client.get(reverse("ui:home")).status_code == 302

    settings.REQUIRE_LOGIN = False
    assert client.get(reverse("ui:home")).status_code == 200


def test_a_stranger_is_sent_to_the_login_page(client, hosted):
    resp = client.get(reverse("ui:home"))

    assert resp.status_code == 302
    assert resp["Location"].startswith(reverse("ui:login"))


def test_the_page_they_wanted_is_carried_across_the_login(client, hosted):
    """Shared links should land on their experiment."""
    target = reverse("ui:appearance")
    resp = client.get(target)

    assert f"next={target}" in resp["Location"]


def test_signing_in_returns_them_to_it(client, hosted, user):
    target = reverse("ui:appearance")

    resp = client.post(reverse("ui:login"),
                       {"username": "ana", "password": "pw", "next": target})

    assert resp.status_code == 302
    assert resp["Location"] == target


def test_a_signed_in_user_sees_the_app(client, hosted, user):
    client.force_login(user)
    assert client.get(reverse("ui:home")).status_code == 200


def test_a_wrong_password_says_so_rather_than_signing_in(client, hosted, user):
    resp = client.post(reverse("ui:login"),
                       {"username": "ana", "password": "not-it"})

    assert resp.status_code == 200
    assert not resp.wsgi_request.user.is_authenticated


def test_signing_out_needs_a_post(client, hosted, user):
    """An `<img>` tag on another site can trigger a GET logout."""
    client.force_login(user)
    assert client.get(reverse("ui:logout")).status_code == 405


def test_signing_out_ends_the_session(client, hosted, user):
    client.force_login(user)

    client.post(reverse("ui:logout"))

    assert client.get(reverse("ui:home")).status_code == 302


def test_the_healthcheck_answers_without_a_session(client, hosted):
    """A 302 healthcheck would make the orchestrator restart a healthy instance
    forever."""
    resp = client.get(reverse("ui:healthz"))

    assert resp.status_code == 200
    assert resp.content == b"ok"


def test_the_language_route_is_reachable_before_signing_in(client, hosted):
    """`set_language` must work before sign-in because the login page carries
    the switcher.
    """
    resp = client.post(reverse("set_language"),
                       {"language": "en", "next": reverse("ui:login")})

    assert resp.status_code == 302
    assert client.get(reverse("ui:login")).status_code == 200


def test_the_url_space_has_no_unintended_exemptions(hosted):
    """The wall is default-closed, so every exemption listed here is deliberate."""
    exempt = set()
    for pattern in get_resolver().url_patterns:
        for route in getattr(pattern, "url_patterns", [pattern]):
            callback = getattr(route, "callback", None)
            if callback is not None and getattr(callback, "login_required", True) is False:
                exempt.add(route.name)

    assert exempt == {"healthz", "set_language", "login"}
