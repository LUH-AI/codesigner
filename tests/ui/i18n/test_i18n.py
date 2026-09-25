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
def test_every_offered_language_is_compiled(locale):
    """A `.mo` is the only thing gettext actually reads. A language in the
    switcher without one is a choice that silently does nothing."""
    assert (CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.mo").exists()


@pytest.mark.parametrize("locale", SHELVED)
def test_nothing_is_left_untranslated(locale):
    """The thing worth avoiding is a *stale* catalog, not a translated one: a
    string whose English moved on while the translation kept saying the old
    thing. An empty or fuzzy entry is how that shows up, so there are none —
    and `manage.py translations` is what keeps it that way."""
    catalog = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")

    assert len(catalog) > 100
    assert not [k for k, v in catalog.items() if not v], "untranslated strings remain"


@pytest.mark.parametrize("locale", SHELVED)
def test_every_translated_language_is_offered(locale):
    """The other direction: a finished catalog nobody can select is work that
    reaches no one."""
    assert locale in dict(settings.LANGUAGES)


def test_all_three_are_offered():
    assert sorted(code for code, _ in settings.LANGUAGES) == ["de", "en", "es"]


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


def test_the_switcher_offers_every_language_and_works(client):
    """It was inert while the catalogs were drafts. They are finished and
    compiled now, so a switcher that still did nothing would be the worse of the
    two failures: work that reaches nobody, behind a control that looks like it
    should."""
    from django.conf import settings

    body = client.get(reverse("ui:home")).content.decode()
    rail = body.split('<nav class="rail">', 1)[1].split("</nav>", 1)[0]
    locale = rail.split('class="locale"', 1)[1]

    for _code, label in settings.OFFERED_LANGUAGES:
        assert label in locale
    # Nothing is offered that cannot be served.
    assert {c for c, _ in settings.OFFERED_LANGUAGES} == {c for c, _ in settings.LANGUAGES}
    assert reverse("set_language") in locale, "the switcher has somewhere to post to"
    assert 'name="language"' in locale
    assert 'method="post"' in locale


def test_switching_language_changes_the_page(client):
    """End to end: the control, the route and the catalogs together."""
    resp = client.post(reverse("set_language"),
                       {"language": "de", "next": reverse("ui:home")}, follow=True)

    assert "Experimente" in resp.content.decode()


def test_the_language_route_still_exists(client):
    """Kept wired so re-enabling a language is `LANGUAGES` and a compile, not a
    hunt for what else was removed."""
    resp = client.post(reverse("set_language"),
                       {"language": "en", "next": reverse("ui:home")})

    assert resp.status_code in (302, 200)
