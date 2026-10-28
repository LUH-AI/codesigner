"""A cookie keeps the browser's theme, and every page carries it as `data-theme`
on `<html>`. Unknown values are ignored.
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
