"""Internationalisation, while German and Spanish are shelved.

The interface is marked for translation throughout and only English is offered.
The de/es catalogs are still in `locale/`, but everything in them that was not
carried over from InteractiveHPO is marked fuzzy — a draft nobody who speaks the
language has read — and there are no compiled `.mo` files, so none of it can
reach a user. An unreviewed translation is worse than an English one; a missing
translation is not.

So what is pinned here is no longer completeness. It is that the machinery stays
intact and inert: the markup keeps working as identity, the switcher does not
offer a language that has nothing behind it, and nothing unreviewed is compiled.
Re-enabling is `LANGUAGES` plus `compilemessages`, once the drafts have been
read.

The one assertion kept from before is the parity check against InteractiveHPO —
those 32 strings *were* reviewed by a person, and if one ever diverges from the
reference app that is still worth knowing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.conf import settings
from django.urls import reverse

CODESIGNER_LOCALE = Path(settings.BASE_DIR) / "locale"
# InteractiveHPO lives beside the codesigner repo; its catalog is the oracle.
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
    """{msgid: msgstr} for every entry with a non-empty msgstr, fuzzy included.

    A minimal parser covering the subset of PO syntax our catalogs use:
    single- and multi-line msgid/msgstr, ignoring the header (empty msgid).
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


# --- the drafts stay drafts ---------------------------------------------------


@pytest.mark.parametrize("locale", SHELVED)
def test_nothing_unreviewed_is_compiled(locale):
    """A `.mo` is the only thing gettext actually reads. While the catalogs hold
    unreviewed drafts there must not be one, or the drafts are live."""
    assert not (CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.mo").exists()


@pytest.mark.parametrize("locale", SHELVED)
def test_the_drafts_are_still_there_to_come_back_to(locale):
    """Shelved, not discarded. Deleting them would mean redoing the work when
    the interface settles, and the reviewed ones would go with it."""
    catalog = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")

    assert len(catalog) > 100


@pytest.mark.parametrize("locale", SHELVED)
def test_a_shelved_language_is_not_offered(locale):
    """The switcher must not list a language with nothing behind it — choosing
    it would silently do nothing at all."""
    assert locale not in dict(settings.LANGUAGES)


def test_english_is_what_is_offered():
    assert [code for code, _ in settings.LANGUAGES] == ["en"]


# --- parity with the reference app, for the strings a person did check --------


@pytest.mark.skipif(
    not IHPO_LOCALE.exists(),
    reason="InteractiveHPO reference checkout not found beside codesigner",
)
@pytest.mark.parametrize("locale", SHELVED)
def test_translations_carried_over_still_match_interactivehpo(locale):
    """For every msgid codesigner shares with InteractiveHPO, the translation
    must be byte-identical to the reference app's — the msgid is the English
    source in both, so a shared string must translate the same. These are the
    only entries in the catalog anyone has verified."""
    ours = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")
    oracle = _parse_po(IHPO_LOCALE / locale / "LC_MESSAGES" / "app.po")

    differing = {
        msgid: (ours[msgid], oracle[msgid])
        for msgid in set(ours) & set(oracle)
        if ours[msgid] != oracle[msgid]
    }

    assert not differing, f"[{locale}] diverged from the reference app: {differing}"


# --- the markup still works ---------------------------------------------------


def test_the_interface_renders_its_english_source(client):
    """`{% translate %}` with no catalog behind it is the identity function, so
    every marked string renders as written. This is what makes leaving strings
    untranslated cost nothing."""
    body = client.get(reverse("ui:home")).content.decode()

    assert "Experiments" in body
    assert "Use the sidebar to create a new experiment." in body


def test_the_switcher_is_on_the_page_and_does_nothing(client):
    """It shows the languages the interface is going to offer, so its place on
    the rail is settled before the catalogs are.

    Inert on purpose, and this is what says so: no form around it, so there is
    nothing for it to submit. The German and Spanish drafts have not been read
    by anyone who speaks them, and half a translation reaching a user is worse
    than none — see OFFERED_LANGUAGES against LANGUAGES in config/settings.py.
    """
    from django.conf import settings

    body = client.get(reverse("ui:home")).content.decode()
    rail = body.split('<nav class="rail">', 1)[1].split("</nav>", 1)[0]
    locale = rail.split('class="locale"', 1)[1]

    for _code, label in settings.OFFERED_LANGUAGES:
        assert label in locale
    assert len(settings.OFFERED_LANGUAGES) > len(settings.LANGUAGES)
    assert "<form" not in locale
    assert "set_language" not in body


def test_the_language_route_still_exists(client):
    """Kept wired so re-enabling a language is `LANGUAGES` and a compile, not a
    hunt for what else was removed."""
    resp = client.post(reverse("set_language"),
                       {"language": "en", "next": reverse("ui:home")})

    assert resp.status_code in (302, 200)
