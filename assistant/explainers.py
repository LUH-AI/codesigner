"""`explainers` reads one Markdown file per explainer from
assistant/knowledge/explainers/, in a folder per kind (explainers/README.md).
"""

import re
from functools import cache
from pathlib import Path

KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"
ROOT = KNOWLEDGE / "explainers"

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _front(text: str) -> dict:
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
    """`read` returns fields and sections for each file of *kind*."""
    found = {}
    for path in sorted((ROOT / kind).glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        match = _FRONT.match(text)
        found[path.stem] = {"fields": _front(match.group(1)),
                            "sections": _sections(text[match.end():])}
    return found
