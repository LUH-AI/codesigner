"""`translations` runs `makemessages`, then regroups each catalog into sections
with a banner per section, since `makemessages` scatters strings that live in
several files. Entry order in a `.po` carries no meaning, and the regrouping
must run on every regeneration or `makemessages` restores source order.
"""
import re
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand

#: The English catalog is authoritative and gathers every interface string into
#: one grouped file. Each `msgstr` mirrors its `msgid` and is compiled, so
#: editing a `msgstr` changes the page while the `msgid` stays the lookup key.
SOURCE_LOCALE = "en"

#: `SECTIONS` lists sections in file order, and the first matching prefix wins,
#: so specific paths precede general ones. Figures motivate the grouping, since
#: `catalog.py` holds their titles and their templates hold their controls.
SECTIONS = (
    ("Site chrome — navigation, base template, account", (
        "ui/templates/ui/base.html", "ui/navigation.py", "access/", "ui/templates/ui/account",
    )),
    ("Figures — titles and descriptions (ui/figures/catalog.py)", (
        "ui/figures/",
    )),
    ("Figures — the figures themselves", (
        "ui/templates/ui/figures/",
    )),
    ("The experiment page, outside the figures", (
        "ui/templates/ui/experiment_detail.html", "ui/templates/ui/experiment_prerun.html",
        "ui/templates/ui/_side",
        "ui/templates/ui/_",
    )),
    ("Optimizer and model settings", (
        "ui/optimizer_labels.py", "ui/forms.py",
        "ui/templates/ui/experiment_settings", "ui/templates/ui/default_experiment",
    )),
    ("Panels — groups, jobs, site administration", (
        "ui/templates/ui/panels/", "ui/views/groups.py", "ui/views/site.py",
    )),
    ("Views — messages, warnings, refusals", (
        "ui/views/", "experiments/", "core/",
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
        # Included unless `--locale` was given.
        if not options["locales"] and SOURCE_LOCALE not in locales:
            locales = [SOURCE_LOCALE] + list(locales)

        # `_apply_wording` runs first because it rewrites the source that
        # `makemessages` then reads. An override left only in the English
        # catalog keeps the stale `msgid` that the German and Spanish catalogs
        # are keyed on.
        if not options["no_apply"] and not options["no_regenerate"]:
            for line in self._apply_wording():
                self.stdout.write(line)

        if not options["no_regenerate"]:
            # `--no-obsolete` drops strings gone from the source, keeping counts
            # honest.
            call_command("makemessages", locale=locales, no_obsolete=True)

        for code in locales:
            path = Path(settings.LOCALE_PATHS[0]) / code / "LC_MESSAGES" / "django.po"
            if not path.is_file():
                continue
            note = " (the English strings)" if code == SOURCE_LOCALE else ""
            self.stdout.write(f"{code}: {self._regroup(path, code)}{note}")

        # `compilemessages` runs here, because a catalog edit must reach the
        # page. `msgfmt` leaves fuzzy entries out of the `.mo`, so a string
        # whose English moved falls back to English.
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

        # Old banners are stripped by their `#. == ` marker, which keeps
        # translators' own comments.
        cleaned = []
        for block in entries:
            lines = [ln for ln in block.split("\n") if not ln.startswith("#. == ")]
            if code == SOURCE_LOCALE:
                # A fuzzy entry means the source moved under an existing
                # `msgstr`, and the newer source wins. A stale catalog line
                # would otherwise override a template edit.
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
        """`_apply_wording` writes English reworded in the catalog back into its
        source files, so the next `makemessages` re-keys every catalog on the
        new text. It edits only files the `#:` references name and reports
        anything it cannot place.
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
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)
        joined = "".join(parts)
        return (joined.replace('\\"', '"').replace("\\n", "\n")
                      .replace("\\t", "\t").replace("\\\\", "\\"))

    @staticmethod
    def _rewrite(refs, old: str, new: str) -> bool:
        """`_rewrite` searches the referenced files instead of trusting line
        numbers, since the `#:` lines predate the `makemessages` that follows
        and may be stale. It writes only when a file holds a single occurrence.
        """
        placed = False
        for ref in dict.fromkeys(r.rpartition(":")[0] for r in refs):
            path = Path(ref)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            # Template extraction doubles every `%`, so the doubling is undone
            # for templates only, since Python strings may hold their own `%%`.
            if path.suffix == ".html":
                old, new = old.replace("%%", "%"), new.replace("%%", "%")
            # The quotes keep "Best sampled" from matching inside "Best sampled,
            # unweighted".
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
        """`_mirror` fills an empty `msgstr` with its `msgid`, or refills it
        under *reset*, which makes the catalog authoritative. A `msgstr` that
        differs from its `msgid` is someone's wording and stays. Plural entries
        take form 0 from `msgid` and form 1 from `msgid_plural`, which suits
        English's two forms.
        """
        if "msgid_plural" in block:
            plural = re.search(
                r'(?ms)^msgid ((?:".*"\n?)+)^msgid_plural ((?:".*"\n?)+)'
                r'^msgstr\[0\] ((?:".*"\n?)+)^msgstr\[1\] ((?:".*"\n?)+)',
                block + "\n")
            if not plural:
                return block
            filled = (plural.group(3).strip() not in ('""', "")
                      and plural.group(4).strip() not in ('""', ""))
            if filled and not reset:
                return block
            return (block[:plural.start(3) - len("msgstr[0] ")]
                    + "msgstr[0] " + plural.group(1).rstrip("\n") + "\n"
                    + "msgstr[1] " + plural.group(2)).rstrip("\n")

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
        """`_explain` replaces gettext's boilerplate header, including its `#,
        fuzzy`, with a note for someone looking for a string.
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
        return len(SECTIONS) - 1

    @classmethod
    def _within(cls, block: str):
        """`_within` gives each block a file-then-line sort key."""
        reference = cls._first_reference(block)
        file, _, line = reference.rpartition(":")
        return (file or reference, int(line) if line.isdigit() else 0)

    @staticmethod
    def _first_reference(block: str) -> str:
        for line in block.split("\n"):
            if line.startswith("#: "):
                return line[3:].split()[0]
        return "~"                        # unreferenced sorts last
