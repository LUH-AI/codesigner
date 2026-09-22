"""What a signed-in person can do about their own account.

*What:* change the password they are signed in with, and read what they are
allowed to do here. Neither existed: a password could only be changed by an
administrator in the Django admin, and nothing anywhere answered "what am I
allowed to do", which now takes assembling four separate grants.

*How:* through the pages, as a signed-in user. Password *reset* is deliberately
not covered because it is deliberately not built — it mails a link to somebody
who cannot sign in, so it needs a mail server; changing a password you already
know does not, which is why one is here and the other is not.
"""

import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    """An instance with accounts."""
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


# ── changing your own password ───────────────────────────────────────────────

def test_a_signed_in_user_can_change_their_own_password(client, hosted, ana,
                                                        django_user_model):
    client.force_login(ana)

    resp = client.post(reverse("access:password_change"), {
        "old_password": "old-pw-12345",
        "new_password1": "a-new-one-98765",
        "new_password2": "a-new-one-98765",
    })

    assert resp.status_code == 302
    assert django_user_model.objects.get(pk=ana.pk).check_password("a-new-one-98765")


def test_the_current_password_is_required(client, hosted, ana, django_user_model):
    """So a borrowed, still-signed-in browser cannot lock its owner out."""
    client.force_login(ana)

    resp = client.post(reverse("access:password_change"), {
        "old_password": "not-the-right-one",
        "new_password1": "a-new-one-98765",
        "new_password2": "a-new-one-98765",
    })

    assert resp.status_code == 200, "it should re-render with the error"
    assert django_user_model.objects.get(pk=ana.pk).check_password("old-pw-12345")


def test_the_configured_validators_still_apply(client, hosted, ana, django_user_model):
    """Django's own form enforces AUTH_PASSWORD_VALIDATORS, and the page shows
    what it said rather than a message of ours guessing which one failed."""
    client.force_login(ana)

    resp = client.post(reverse("access:password_change"), {
        "old_password": "old-pw-12345",
        "new_password1": "pw", "new_password2": "pw",
    })

    assert resp.status_code == 200
    assert django_user_model.objects.get(pk=ana.pk).check_password("old-pw-12345")


def test_a_stranger_cannot_reach_the_page(client, hosted):
    """It is behind the wall like everything else."""
    resp = client.get(reverse("access:password_change"))

    assert resp.status_code == 302
    assert reverse("access:login") in resp["Location"]


# ── what you are allowed to do ───────────────────────────────────────────────

def test_the_account_page_lists_what_this_user_may_do(client, hosted, ana,
                                                      django_user_model):
    """Assembled from the grants, because there are four of them now and no
    other page can say what they add up to."""
    ana = _granted(django_user_model, ana, "view_all_experiments")
    client.force_login(ana)

    # str() because the labels are lazy translations, which are not dict keys
    # you can look up with a plain string.
    granted = {str(what): allowed
               for what, allowed in client.get(reverse("ui:account")).context["powers"]}

    assert granted["See every experiment on this instance"] is True
    assert granted["Run, edit and delete other people's experiments"] is False
    assert granted["Change the settings every experiment inherits"] is False


def test_a_superuser_is_told_why_everything_is_ticked(client, hosted,
                                                      django_user_model):
    """Django grants them every permission without any of them being a grant, so
    four ticks with no explanation would read as though somebody had ticked
    them."""
    # One of the four is gated by the instance-wide flag, which is off by
    # default, so "everything is ticked" needs the floor raised first. The test
    # below is the one about that floor being legible when it is not.
    hosted.ALLOW_CUSTOM_MODELS = True
    root = django_user_model.objects.create_superuser(username="su", password="pw")
    client.force_login(root)

    resp = client.get(reverse("ui:account"))

    assert resp.context["is_administrator"]
    assert all(allowed for _what, allowed in resp.context["powers"])
    assert "superuser" in resp.content.decode()


def test_the_instance_wide_switch_is_shown_as_such(client, hosted,
                                                   django_user_model):
    """`ALLOW_CUSTOM_MODELS=False` is a floor under every account, so an
    administrator seeing a dash there should not go looking for the grant."""
    hosted.ALLOW_CUSTOM_MODELS = False
    root = django_user_model.objects.create_superuser(username="su", password="pw")
    client.force_login(root)

    resp = client.get(reverse("ui:account"))

    assert resp.context["custom_models_off"]
    assert dict(resp.context["powers"])  # rendered without raising


def test_there_is_no_account_page_without_accounts(client, settings):
    """Every answer on it would be an unconditional yes, which describes the
    instance rather than anybody's account."""
    settings.REQUIRE_LOGIN = False

    assert client.get(reverse("ui:account")).status_code == 404


def test_the_settings_nav_offers_it_only_when_signed_in(client, hosted, ana):
    client.force_login(ana)

    html = client.get(reverse("ui:appearance")).content.decode()

    assert reverse("ui:account") in html
