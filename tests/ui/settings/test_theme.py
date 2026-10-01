"""Light, dark, or as the system is — chosen under Settings → Appearance.

*What:* the choice is kept in a cookie for this browser and carried on every
page as `data-theme` on <html>, which the stylesheet and theme.js read. A
browser that has chosen nothing follows its system, and anything that is not
one of the three choices is ignored rather than trusted.

*How:* through the appearance page and the attribute on the next page drawn.
"""

import pytest
from django.urls import reverse

from ui.context_processors import THEME_COOKIE

pytestmark = pytest.mark.django_db


def _theme(client, path="ui:appearance"):
    html = client.get(reverse(path)).content.decode()
    return html.split("data-theme=\"", 1)[1].split('"', 1)[0]


def test_a_browser_that_has_chosen_nothing_follows_its_system(client):
    assert _theme(client) == "system"


@pytest.mark.parametrize("choice", ["light", "dark", "system"])
def test_the_choice_is_kept_and_carried_on_every_page(client, choice):
    resp = client.post(reverse("ui:appearance"), {"theme": choice})

    assert resp.status_code == 302
    assert resp.cookies[THEME_COOKIE].value == choice
    assert _theme(client, "ui:home") == choice


def test_something_else_is_ignored(client):
    client.post(reverse("ui:appearance"), {"theme": "dark"})

    resp = client.post(reverse("ui:appearance"), {"theme": "</html><script>"})

    assert THEME_COOKIE not in resp.cookies
    assert _theme(client) == "dark"


def test_a_tampered_cookie_falls_back_to_the_system(client):
    client.cookies[THEME_COOKIE] = "neon"

    assert _theme(client) == "system"


def test_the_page_offers_the_three(client):
    html = client.get(reverse("ui:appearance")).content.decode()

    for value in ("light", "dark", "system"):
        assert f'value="{value}"' in html
