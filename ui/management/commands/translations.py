"""Regenerate the translation catalogs, then put them in an order a person can read.

`makemessages` writes entries in the order `xgettext` met them, which is source
order across a sorted file list. That is not a bad order, but it scatters
anything whose strings live in more than one place: a figure's labels are in
`ui/figures/catalog.py` while the rest of its interface is in its own template,
so editing one figure's wording means finding its strings in two distant parts
of a two-thousand-line file.

So this regroups afterwards, into sections, with a banner naming each file. The
order of entries in a `.po` carries no meaning — `msgid` is the key — so this is
free to rearrange them.

It has to be a step of the same command rather than a one-off tidy, because the
next `makemessages` would write the old order back. Run this instead of
`makemessages` and the grouping survives.
"""
import re
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand

#: The source language, which gets a catalog of its own even though there is
#: nothing to translate into it.
#:
#: This is the one place every English string in the interface is listed. Django
#: does not need it — for the source language the `msgid` *is* the string, and
#: the app reads English straight out of the templates — so it exists purely to
#: be read: 443 strings that otherwise live across sixty source files, in one
#: grouped file with a `#:` reference under each saying where it came from.
#:
#: It is also authoritative. Each `msgstr` is filled with its own `msgid` and
#: the catalog is compiled, so `{% translate %}` answers out of it: change the
#: `msgstr` and that is what the page says. The `msgid` stays the source text —
#: it is the lookup key — so the source file is still where a string is born,
#: but the wording can be settled here without going and finding it.
SOURCE_LOCALE = "en"

#: Sections, in the order they appear in the file, each naming the paths that
#: belong to it. First match wins, so the specific comes before the general.
#:
#: The intent is that someone editing one part of the interface finds all of it
#: in one place. Figures are the case that drove it: `catalog.py` holds every
#: figure's title and description, and each figure's own template holds its
#: controls, so they are pulled together here.
SECTIONS = (
    ("Site chrome — navigation, base template, account", (
        "templates/base.html", "ui/navigation.py", "access/", "ui/templates/ui/account",
    )),
    ("Figures — titles and descriptions (ui/figures/catalog.py)", (
        "ui/figures/",
    )),
    ("Figures — the figures themselves", (
        "ui/templates/ui/figures/",
    )),
    ("The experiment page, outside the figures", (
        "ui/templates/ui/experiment_detail.html", "ui/templates/ui/_side",
        "ui/templates/ui/_",
    )),
    ("Optimizer and model settings", (
        "ui/optimizer_labels.py", "ui/forms.py", "ui/registry.py",
        "ui/templates/ui/experiment_settings", "ui/templates/ui/default_experiment",
    )),
    ("Panels — groups, jobs, site administration", (
        "ui/templates/ui/panels/", "ui/panels.py",
    )),
    ("Views — messages, warnings, refusals", (
        "ui/views.py", "ui/services/", "core/",
    )),
    ("Everything else", ()),
)


class Command(BaseCommand):
    help = "Update the translation catalogs and group their strings by area."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-apply", action="store_true",
            help="Do not write reworded English back into the source files.")
        parser.add_argument(
            "--locale", "-l", action="append", dest="locales", default=None,
            help="Language to update; repeatable. Defaults to every catalog present.")
        parser.add_argument(
            "--no-regenerate", action="store_true",
            help="Only regroup what is already there, without calling makemessages.")

    def handle(self, *args, **options):
        locales = options["locales"] or self._present()
        if not locales:
            self.stderr.write("No catalogs under locale/, and no --locale given.")
            return
        # Always, unless someone named locales explicitly and left it out.
        if not options["locales"] and SOURCE_LOCALE not in locales:
            locales = [SOURCE_LOCALE] + list(locales)

        # First, because it rewrites the source that `makemessages` then reads.
        # A wording edited in the English catalog is only an override until it
        # reaches the string it overrides; left there, the `msgid` — which is
        # the key the German and Spanish catalogs are keyed on — keeps saying
        # the old thing forever, and every translator works from text no one has
        # seen on screen in months.
        if not options["no_apply"] and not options["no_regenerate"]:
            for line in self._apply_wording():
                self.stdout.write(line)

        if not options["no_regenerate"]:
            # `--no-obsolete` because a string that no longer exists in the
            # source is not a backlog item, and leaving it in makes the count
            # meaningless.
            call_command("makemessages", locale=locales, no_obsolete=True)

        for code in locales:
            path = Path(settings.LOCALE_PATHS[0]) / code / "LC_MESSAGES" / "django.po"
            if not path.is_file():
                continue
            note = " (the English strings)" if code == SOURCE_LOCALE else ""
            self.stdout.write(f"{code}: {self._regroup(path, code)}{note}")

        # Compiled here rather than left to the reader, because an edit that does
        # not reach the page is the whole failure this guards against. A `.mo` is
        # the only thing gettext reads, and `msgfmt` leaves fuzzy entries out of
        # it — so a string whose English has moved on falls back to English
        # rather than asserting a translation of something no longer said.
        call_command("compilemessages", locale=list(locales), verbosity=0)
        self.stdout.write(f"compiled: {', '.join(locales)}")

    @staticmethod
    def _present():
        root = Path(settings.LOCALE_PATHS[0])
        return sorted(p.name for p in root.glob("*") if (p / "LC_MESSAGES/django.po").is_file())

    def _regroup(self, path: Path, code: str = "") -> str:
        text = path.read_text(encoding="utf-8")
        blocks = text.split("\n\n")
        header, entries = blocks[0], [b for b in blocks[1:] if b.strip()]
        if code == SOURCE_LOCALE:
            header = self._explain(header)

        # Strip any banner this command wrote last time, so they do not
        # accumulate. They are recognised by their own marker rather than by
        # looking like a comment, because a translator's own notes are comments
        # too and must survive.
        cleaned = []
        for block in entries:
            lines = [ln for ln in block.split("\n") if not ln.startswith("#. == ")]
            if code == SOURCE_LOCALE:
                # A fuzzy entry means the source text moved under an existing
                # `msgstr`. The source is the newer statement of the two, so it
                # wins and the wording here is reset to it — otherwise a
                # template edit would be silently overridden by a stale line in
                # this file, which is the one failure mode of making it
                # authoritative.
                stale = any(re.match(r"^#,.*\bfuzzy\b", ln) for ln in lines)
                lines = [ln for ln in lines if not re.match(r"^#,.*\bfuzzy\b", ln)]
                block = self._mirror("\n".join(lines), reset=stale)
                if block.strip():
                    cleaned.append(block)
                continue
            if any(ln.strip() for ln in lines):
                cleaned.append("\n".join(lines))

        ordered, counts = [], []
        for index, (title, prefixes) in enumerate(SECTIONS):
            chosen = [b for b in cleaned if self._section(b) == index]
            if not chosen:
                continue
            chosen.sort(key=self._within)
            counts.append(f"{title.split(' — ')[0]} {len(chosen)}")
            banner = f"#. == {title} " + "=" * max(4, 66 - len(title))
            ordered.append(banner + "\n" + chosen[0])
            ordered.extend(chosen[1:])

        path.write_text("\n\n".join([header] + ordered) + "\n", encoding="utf-8")
        return f"{len(cleaned)} strings, {len(counts)} sections"

    def _apply_wording(self):
        """Push English reworded in the catalog back into the files it came from.

        An entry whose `msgstr` differs from its `msgid` is somebody having
        settled the wording here rather than hunting for the template. This
        carries it home: the source gets the new text, and the next
        `makemessages` makes it the `msgid`, so the two agree again and the
        other catalogs are re-keyed on what the screen actually says.

        Conservative on purpose. It only touches the exact lines the `#:`
        references name, and only where the old text is there to be replaced —
        anything it cannot place is reported rather than guessed at, because
        the cost of a wrong guess is a corrupted source file.
        """
        path = Path(settings.LOCALE_PATHS[0]) / SOURCE_LOCALE / "LC_MESSAGES" / "django.po"
        if not path.is_file():
            return []

        done, missed = [], []
        for block in path.read_text(encoding="utf-8").split("\n\n")[1:]:
            if "msgid_plural" in block:
                continue
            match = re.search(r'(?ms)^msgid ((?:".*"\n?)+)^msgstr ((?:".*"\n?)+)', block + "\n")
            if not match:
                continue
            old, new = self._unquote(match.group(1)), self._unquote(match.group(2))
            if not new or old == new:
                continue
            refs = [r for line in block.split("\n") if line.startswith("#: ")
                    for r in line[3:].split()]
            if self._rewrite(refs, old, new):
                done.append((old, new))
            else:
                missed.append((old, new, refs))

        out = []
        for old, new in done:
            out.append(f"  reworded: {old[:44]!r} -> {new[:44]!r}")
        for old, new, refs in missed:
            out.append(f"  COULD NOT PLACE {old[:40]!r} (looked in {', '.join(refs) or 'nowhere'});"
                       f" edit the source directly, or the override stays an override")
        if out:
            out.insert(0, f"{len(done)} reworded in place"
                          + (f", {len(missed)} not placed" if missed else ""))
        return out

    @staticmethod
    def _unquote(chunk: str) -> str:
        """The text of a `msgid`/`msgstr`, joined and unescaped."""
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)
        joined = "".join(parts)
        return (joined.replace('\\"', '"').replace("\\n", "\n")
                      .replace("\\t", "\t").replace("\\\\", "\\"))

    @staticmethod
    def _rewrite(refs, old: str, new: str) -> bool:
        """Replace *old* with *new* in the files the references name.

        Searched rather than addressed by line number. The `#:` line is only
        as fresh as the last `makemessages`, and this runs *before* that one —
        so by here the numbers are a generation behind and point at whatever
        has since moved into that row. A stale number is not a near miss; it is
        an edit to an unrelated line.

        Written only on a single unambiguous occurrence. Two matches in a file
        means the catalog cannot say which one was meant, and a translation
        catalog is no reason to guess at editing source.
        """
        placed = False
        for ref in dict.fromkeys(r.rpartition(":")[0] for r in refs):
            path = Path(ref)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            # Extracting a template doubles every `%`, to protect the
            # `%`-formatting `blocktranslate` interpolates its own variables
            # with. The source never had the double, so undo it — but only for
            # the templates that caused it, since a Python string may hold a
            # `%%` of its own.
            if path.suffix == ".html":
                old, new = old.replace("%%", "%"), new.replace("%%", "%")
            # Matched with its quotes, which is what makes it a string literal
            # rather than a run of words. Without them "Best sampled" also
            # matches inside "Best sampled, unweighted" — two hits, so the
            # ambiguity guard below refuses a rewording that was never
            # ambiguous.
            for quote in ('"', "'"):
                candidate = quote + old.replace(quote, "\\" + quote) + quote
                if text.count(candidate) != 1:
                    continue
                replacement = quote + new.replace(quote, "\\" + quote) + quote
                path.write_text(text.replace(candidate, replacement, 1),
                                encoding="utf-8")
                placed = True
                break
        return placed

    @staticmethod
    def _mirror(block: str, reset: bool) -> str:
        """Give an entry a `msgstr` equal to its `msgid`, where it has none.

        This is what makes the file authoritative rather than a listing: an
        empty `msgstr` means "no translation" and gettext answers with the
        `msgid`, so nothing here could ever change the page. Filled, it is what
        the page says.

        Only where it is empty, or where *reset* says the source moved. An
        `msgstr` that differs from its `msgid` is somebody's wording and is left
        exactly alone.

        Plural entries are skipped — `msgstr[0]`/`msgstr[1]` need a rule per
        language and there are two of them in the whole catalog, so they keep
        gettext's own fallback to the source.
        """
        if "msgid_plural" in block:
            return block

        match = re.search(r'(?ms)^msgid ((?:".*"\n?)+)^msgstr ((?:".*"\n?)+)', block + "\n")
        if not match:
            return block
        msgid, msgstr = match.group(1), match.group(2)
        already = msgstr.strip() not in ('""', "")
        if already and not reset:
            return block
        return (block[:match.start(2) - len("msgstr ")]
                + "msgstr " + msgid).rstrip("\n")

    @staticmethod
    def _explain(header: str) -> str:
        """Say what this file is, in place of gettext's boilerplate.

        A translator opening a `.po` knows what one is. This catalog is for
        someone looking for a string, and the first thing they see should not be
        SOME DESCRIPTIVE TITLE and FIRST AUTHOR. The `#, fuzzy` goes too: on a
        header it means "not filled in", and this one is.
        """
        keep = [ln for ln in header.split("\n")
                if not ln.startswith("#") or ln.startswith('"')]
        return "\n".join([
            "# Every English string in the interface, in one file.",
            "#",
            "# Generated by `manage.py translations` — run that after changing any",
            "# wording, and it rewrites this from the source.",
            "#",
            "# THIS is where to edit the wording. Change a `msgstr` and run",
            "# `manage.py translations` — it recompiles, and the page says what you",
            "# wrote. Leave the `msgid` alone: it is the key the code looks a string",
            "# up by, and it mirrors the source exactly.",
            "#",
            "# The `#:` line under each entry says which file and line the string was",
            "# born in. Editing there works too, and is what changes the `msgid`; if",
            "# you do, the wording here is reset to the source on the next run, since",
            "# the source is then the newer of the two.",
            "#",
            "# The other catalogs, locale/de and locale/es, are translations and",
            "# behave normally.",
            "#",
            "# Grouped into sections; search for `#. == ` to move between them.",
        ] + keep)

    @classmethod
    def _section(cls, block: str) -> int:
        reference = cls._first_reference(block)
        for index, (_title, prefixes) in enumerate(SECTIONS):
            if any(reference.startswith(p) for p in prefixes):
                return index
        return len(SECTIONS) - 1          # "Everything else"

    @classmethod
    def _within(cls, block: str):
        """File, then line, so a group reads in the order it is written."""
        reference = cls._first_reference(block)
        file, _, line = reference.rpartition(":")
        return (file or reference, int(line) if line.isdigit() else 0)

    @staticmethod
    def _first_reference(block: str) -> str:
        for line in block.split("\n"):
            if line.startswith("#: "):
                return line[3:].split()[0]
        return "~"                        # no reference sorts last within its section
