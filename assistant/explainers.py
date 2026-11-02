"""The explainers the assistant reads, from assistant/knowledge/explainers*/:
one Markdown file each, with frontmatter and `## ` sections, in a folder per
kind (pages, tradeoffs, …). A file in `explainers/` replaces the draft of the
same id in `explainers-DRAFT/`; the drafts are in use until they are reviewed
(explainers-DRAFT/README.md).
"""

import re
from functools import cache
from pathlib import Path

KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"
ROOTS = (KNOWLEDGE / "explainers", KNOWLEDGE / "explainers-DRAFT")

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _front(text: str) -> dict:
    """The frontmatter's fields: a list for `[a, b]` or `  - ` items, else text."""
    fields, current = {}, None
    for line in text.splitlines():
        item = re.match(r"^  - (.*)$", line)
        if item and current:
            fields[current].append(item.group(1).strip())
            continue
        key, _, value = line.partition(":")
        current, value = key.strip(), value.strip()
        if value.startswith("["):
            fields[current] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            fields[current] = value or []
    return fields


def _sections(body: str) -> dict:
    parts = re.split(r"^## (.+)$", _COMMENT.sub("", body), flags=re.M)
    return {title.strip(): text.strip() for title, text in zip(parts[1::2], parts[2::2])}


@cache
def read(kind: str) -> dict:
    """The explainers of *kind* (the folder's name) by file stem, each
    `{"fields": frontmatter, "sections": {title: text}}`; the reviewed one of
    an id first, its draft skipped."""
    found, ids = {}, set()
    for root in ROOTS:
        for path in sorted((root / kind).glob("*.md")):
            if path.name == "index.md":
                continue
            text = path.read_text(encoding="utf-8")
            match = _FRONT.match(text)
            fields = _front(match.group(1))
            if fields["id"] in ids:
                continue
            ids.add(fields["id"])
            found[path.stem] = {"fields": fields, "sections": _sections(text[match.end():])}
    return found
