"""Step 8: internationalisation (English / German / Spanish).

Codesigner reuses InteractiveHPO's human-verified de/es translations. Because
the Streamlit app keys its catalog on the *English* text (``_STRINGS`` values
are the gettext msgids), and Django's ``{% translate %}`` / ``gettext`` also
key on the English source string, a translation that is correct in the
Streamlit app is correct here verbatim — the msgid is identical.

These tests pin four things:

* every English string marked for translation has a non-empty de and es
  translation (the CI completeness check the plan calls for, adapted from
  ``utils/check_translations.py``);
* for every msgid codesigner shares with InteractiveHPO, codesigner's
  translation *equals* the oracle's — a direct parity assertion against the
  reference app rather than a hand-copied table (skipped if the sibling
  InteractiveHPO checkout isn't present);
* the default language renders English;
* posting to ``set_language`` switches the active language and the rendered
  page comes back translated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.conf import settings
from django.urls import reverse
from django.utils import translation

# --- locating the two catalogs ------------------------------------------------

CODESIGNER_LOCALE = Path(settings.BASE_DIR) / "locale"
# InteractiveHPO lives beside the codesigner repo; its catalog is the oracle.
IHPO_LOCALE = Path(settings.BASE_DIR).parent / "InteractiveHPO" / "locale"

NON_ENGLISH = ["de", "es"]


def _unquote(s: str) -> str:
    return (
        s[1:-1]
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _parse_po(path: Path) -> dict[str, str]:
    """Return {msgid: msgstr} for every translated (non-empty msgstr) entry.

    A minimal parser covering the subset of PO syntax our catalogs use:
    single- and multi-line msgid/msgstr, ignoring the header (empty msgid)
    and any fuzzy/comment lines.
    """
    catalog: dict[str, str] = {}
    msgid: str | None = None
    msgstr: str | None = None
    in_msgstr = False

    def _flush():
        if msgid and msgstr:  # skip header (msgid "") and untranslated entries
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


def _codesigner_msgids() -> set[str]:
    """Every msgid makemessages extracted for German (the set of strings the
    app actually marks for translation; identical across locales)."""
    po = _parse_po_all(CODESIGNER_LOCALE / "de" / "LC_MESSAGES" / "django.po")
    return set(po)


def _parse_po_all(path: Path) -> dict[str, str]:
    """Like _parse_po but keeps entries whose msgstr is empty too, so callers
    can distinguish 'string is marked' from 'string is translated'."""
    catalog: dict[str, str] = {}
    msgid: str | None = None
    msgstr: str | None = None
    in_msgstr = False

    def _flush():
        if msgid:  # keep any real msgid, translated or not; skip header
            catalog[msgid] = msgstr or ""

    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
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


# --- completeness -------------------------------------------------------------


def _fuzzy_msgids(path: Path) -> set[str]:
    """msgids flagged '#, fuzzy' (excluding the header). makemessages marks a
    guessed translation fuzzy; gettext then refuses to compile it, so at runtime
    the string silently falls back to English — as bad as no translation."""
    fuzzy: set[str] = set()
    pending = False
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#,") and "fuzzy" in line:
            pending = True
        elif line.startswith("msgid "):
            mid = _unquote(line[len("msgid "):].strip())
            if pending and mid:  # skip the header (empty msgid)
                fuzzy.add(mid)
            pending = False
    return fuzzy


@pytest.mark.parametrize("locale", NON_ENGLISH)
def test_every_marked_string_is_translated(locale):
    """Adapts utils/check_translations.py: every msgid the app marks for
    translation must have a non-empty, non-fuzzy msgstr in each non-English
    catalog, so switching language never falls back to English for a visible
    string. Fuzzy entries count as untranslated — gettext won't compile them."""
    po_path = CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po"
    assert po_path.exists(), f"missing catalog: {po_path}"

    entries = _parse_po_all(po_path)
    untranslated = sorted(mid for mid, mstr in entries.items() if not mstr)
    assert not untranslated, (
        f"[{locale}] {len(untranslated)} untranslated string(s): {untranslated}"
    )

    fuzzy = sorted(_fuzzy_msgids(po_path))
    assert not fuzzy, f"[{locale}] {len(fuzzy)} fuzzy (uncompiled) string(s): {fuzzy}"


# --- parity against the InteractiveHPO oracle ---------------------------------


@pytest.mark.skipif(
    not IHPO_LOCALE.exists(),
    reason="InteractiveHPO reference checkout not found beside codesigner",
)
@pytest.mark.parametrize("locale", NON_ENGLISH)
def test_translations_match_interactivehpo_reference(locale):
    """For every msgid codesigner shares with InteractiveHPO, codesigner's
    translation must be byte-identical to the reference app's — the msgid is
    the English source in both, so a shared string must translate the same."""
    ours = _parse_po(CODESIGNER_LOCALE / locale / "LC_MESSAGES" / "django.po")
    oracle = _parse_po(IHPO_LOCALE / locale / "LC_MESSAGES" / "app.po")

    shared = set(ours) & set(oracle)
    assert shared, "expected codesigner to reuse at least some reference strings"

    mismatches = {
        mid: (ours[mid], oracle[mid]) for mid in shared if ours[mid] != oracle[mid]
    }
    assert not mismatches, (
        f"[{locale}] translations diverge from InteractiveHPO: {mismatches}"
    )


# --- runtime behaviour --------------------------------------------------------


def test_gettext_translates_a_known_string_to_german():
    """The gettext machinery is wired (LOCALE_PATHS + compiled .mo): activating
    German returns the German source for a representative UI string."""
    with translation.override("de"):
        assert translation.gettext("Number of trials") == "Anzahl der Versuche"


def test_gettext_translates_a_known_string_to_spanish():
    """Same wiring check for Spanish."""
    with translation.override("es"):
        assert translation.gettext("Number of trials") == "Número de pruebas"


def test_default_language_renders_english(client):
    """With no language chosen the sidebar renders the English source text."""
    resp = client.get(reverse("web:home"))
    assert resp.status_code == 200
    body = resp.content.decode()
    assert "Experiments" in body
    assert "Use the sidebar to create a new experiment." in body


def test_language_switcher_lists_all_three_languages(client):
    """The sidebar switcher offers English, German and Spanish."""
    body = client.get(reverse("web:home")).content.decode()
    for label in ("English", "Deutsch", "Español"):
        assert label in body


@pytest.mark.parametrize(
    "code, expected",
    [
        ("de", "Erstellen Sie über die Seitenleiste"),
        ("es", "Usa la barra lateral"),
    ],
)
def test_set_language_switches_rendered_page(client, code, expected):
    """Posting to set_language activates the locale for the session, and the
    next page render comes back translated (not the English source)."""
    home = reverse("web:home")
    resp = client.post(
        reverse("set_language"),
        {"language": code, "next": home},
    )
    assert resp.status_code in (302, 200)

    body = client.get(home).content.decode()
    assert expected in body
    assert "Use the sidebar to create a new experiment." not in body
