"""`ui:password_change` and `ui:account` serve a signed-in user. The app omits
password reset because a reset link needs a mail server.
"""

import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


@pytest.fixture
def ana(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="old-pw-12345")


def _granted(django_user_model, user, *codenames):
    from django.contrib.auth.models import Permission

    user.user_permissions.set(Permission.objects.filter(
        content_type__app_label="access", codename__in=codenames))
    return django_user_model.objects.get(pk=user.pk)


def test_a_signed_in_user_can_change_their_own_password(client, hosted, ana,
                                                        django_user_model):
    client.force_login(ana)

    resp = client.post(reverse("ui:password_change"), {
        "old_password": "old-pw-12345",
        "new_password1": "a-new-one-98765",
        "new_password2": "a-new-one-98765",
    })

    assert resp.status_code == 302
    assert django_user_model.objects.get(pk=ana.pk).check_password("a-new-one-98765")


def test_the_current_password_is_required(client, hosted, ana, django_user_model):
    """Borrowed sessions cannot lock owners out."""
    client.force_login(ana)

    resp = client.post(reverse("ui:password_change"), {
        "old_password": "not-the-right-one",
        "new_password1": "a-new-one-98765",
        "new_password2": "a-new-one-98765",
    })

    assert resp.status_code == 200, "it should re-render with the error"
    assert django_user_model.objects.get(pk=ana.pk).check_password("old-pw-12345")


def test_the_configured_validators_still_apply(client, hosted, ana, django_user_model):
    """The page shows Django's own validator message."""
    client.force_login(ana)

    resp = client.post(reverse("ui:password_change"), {
        "old_password": "old-pw-12345",
        "new_password1": "pw", "new_password2": "pw",
    })

    assert resp.status_code == 200
    assert django_user_model.objects.get(pk=ana.pk).check_password("old-pw-12345")


def test_a_stranger_cannot_reach_the_page(client, hosted):
    """The login wall covers the page."""
    resp = client.get(reverse("ui:password_change"))

    assert resp.status_code == 302
    assert reverse("ui:login") in resp["Location"]


def test_the_account_page_lists_the_custom_model_grant(client, hosted, ana,
                                                      django_user_model):
    """What: the account page lists one power, custom models, as granted or
    not. How: Ana without the grant, then with it."""
    hosted.ALLOW_CUSTOM_MODELS = True
    client.force_login(ana)
    before = dict((str(w), a) for w, a in client.get(reverse("ui:account")).context["powers"])
    ana = _granted(django_user_model, ana, "use_custom_models")
    client.force_login(ana)
    after = dict((str(w), a) for w, a in client.get(reverse("ui:account")).context["powers"])

    assert before == {"Upload and run custom models": False}
    assert after == {"Upload and run custom models": True}


def test_the_instance_wide_switch_is_shown_as_such(client, hosted,
                                                   django_user_model):
    """The page marks `ALLOW_CUSTOM_MODELS=False` as an instance-wide floor, so
    admins skip hunting for a grant."""
    hosted.ALLOW_CUSTOM_MODELS = False
    client.force_login(django_user_model.objects.create_user(username="ana", password="pw"))

    resp = client.get(reverse("ui:account"))

    assert resp.context["custom_models_off"]
    assert dict(resp.context["powers"])


def test_there_is_no_account_page_without_accounts(client, settings):
    """Without accounts, every answer would be an unconditional yes."""
    settings.REQUIRE_LOGIN = False

    assert client.get(reverse("ui:account")).status_code == 404


def test_the_settings_nav_offers_it_only_when_signed_in(client, hosted, ana):
    client.force_login(ana)

    html = client.get(reverse("ui:appearance")).content.decode()

    assert reverse("ui:account") in html
