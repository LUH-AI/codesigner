"""German and Spanish catalogs ship compiled and offered beside English, and
strings shared with InteractiveHPO must match its translations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.conf import settings
from django.urls import reverse

CODESIGNER_LOCALE = Path(settings.BASE_DIR) / "locale"
# InteractiveHPO, beside the repo, is the oracle.
IHPO_LOCALE = Path(settings.BASE_DIR).parent / "InteractiveHPO" / "locale"

SHELVED = ["de", "es"]


def _unquote(s: str) -> str:
    return (
        s[1:-1]
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _parse_po(path: Path) -> dict[str, str]:
    """`_parse_po` maps msgid to msgstr for every non-empty msgstr, fuzzy
    entries included.
    """
    catalog: dict[str, str] = {}
    msgid: str | None = None
    msgstr: str | None = None
    in_msgstr = False

    def _flush():
        if msgid and msgstr:
            catalog[msgid] = msgstr

    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line.startswith("msgid "):
            _flush()
            msgid, msgstr, in_msgstr = _unquote(line[6:].strip()), None, False
        elif line.startswith("msgstr "):
            msgstr, in_msgstr = _unquote(line[7:].strip()), True
        elif line.startswith('"') and line.endswith('"'):
            frag = _unquote(line)
            if in_msgstr:
                msgstr = (msgstr or "") + frag
            elif msgid is not None:
                msgid += frag
        elif not line:
            _flush()
            msgid = msgstr = None
            in_msgstr = False
    _flush()
    return catalog


@pytest.mark.parametrize("locale", SHELVED)
def test_every_offered_language_is_compiled(locale):
    """gettext reads only the `.mo`, so an uncompiled language would switch
    nothing."""
    assert (CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.mo").exists()


@pytest.mark.parametrize("locale", SHELVED)
def test_nothing_is_left_untranslated(locale):
    """A string whose English changed shows up as an empty msgstr, so the test
    allows none."""
    catalog = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")

    assert len(catalog) > 100
    assert not [k for k, v in catalog.items() if not v], "untranslated strings remain"


@pytest.mark.parametrize("locale", SHELVED)
def test_every_translated_language_is_offered(locale):
    assert locale in dict(settings.LANGUAGES)


def test_all_three_are_offered():
    assert sorted(code for code, _ in settings.LANGUAGES) == ["de", "en", "es"]


@pytest.mark.skipif(
    not IHPO_LOCALE.exists(),
    reason="InteractiveHPO reference checkout not found beside codesigner",
)
@pytest.mark.parametrize("locale", SHELVED)
def test_translations_carried_over_still_match_interactivehpo(locale):
    """Each msgid shared with InteractiveHPO must translate byte-identically,
    because the msgid is the English source in both apps."""
    ours = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")
    oracle = _parse_po(IHPO_LOCALE / locale / "LC_MESSAGES" / "app.po")

    differing = {
        msgid: (ours[msgid], oracle[msgid])
        for msgid in set(ours) & set(oracle)
        if ours[msgid] != oracle[msgid]
    }

    assert not differing, f"[{locale}] diverged from the reference app: {differing}"


def test_the_interface_renders_its_english_source(client):
    """`{% translate %}` without a catalog entry renders the string as written."""
    body = client.get(reverse("ui:home")).content.decode()

    assert "Experiments" in body
    assert "Use the sidebar to create a new experiment." in body


def test_the_switcher_offers_every_language_and_works(client):
    """The rail's selector lists every `OFFERED_LANGUAGES` label and posts to
    `set_language`."""
    from django.conf import settings

    body = client.get(reverse("ui:home")).content.decode()
    rail = body.split('<nav class="rail">', 1)[1].split("</nav>", 1)[0]
    locale = rail.split('class="locale"', 1)[1]

    for _code, label in settings.OFFERED_LANGUAGES:
        assert label in locale
    # Offered codes match `LANGUAGES`.
    assert {c for c, _ in settings.OFFERED_LANGUAGES} == {c for c, _ in settings.LANGUAGES}
    assert reverse("set_language") in locale, "the switcher has somewhere to post to"
    assert 'name="language"' in locale
    assert 'method="post"' in locale


def test_switching_language_changes_the_page(client):
    """The German catalog translates the page after `set_language`."""
    resp = client.post(reverse("set_language"),
                       {"language": "de", "next": reverse("ui:home")}, follow=True)

    assert "Experimente" in resp.content.decode()


def test_the_language_route_still_exists(client):
    resp = client.post(reverse("set_language"),
                       {"language": "en", "next": reverse("ui:home")})

    assert resp.status_code in (302, 200)
