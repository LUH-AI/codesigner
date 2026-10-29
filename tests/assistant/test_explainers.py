"""The assistant's explainers (assistant/knowledge/explainers*/) follow the
template for their kind in `_templates/`: the same frontmatter fields, the same
sections in the same order, and references that point at things that exist."""

import re
from functools import cache
from pathlib import Path

import pytest
from django.conf import settings

KNOWLEDGE = Path(settings.BASE_DIR) / "assistant" / "knowledge"
TEMPLATES = KNOWLEDGE / "explainers-DRAFT" / "_templates"
FOLDERS = {"concepts": "concept", "figures": "figure", "pages": "page"}
ABOUT_KINDS = {"figure", "page", "field", "element", "data-assist", "action"}
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def _explainers():
    found = []
    for root in sorted(KNOWLEDGE.glob("explainers*")):
        for folder in FOLDERS:
            found += [p for p in sorted((root / folder).glob("*.md")) if p.name != "index.md"]
    return found


def _split(text):
    match = FRONT.match(text)
    assert match, "no frontmatter"
    return match.group(1), text[match.end():]


def _fields(front):
    """The frontmatter's fields: lists in brackets or as `  - ` items, else text."""
    fields, current = {}, None
    for line in front.splitlines():
        item = re.match(r"^  - (.*)$", line)
        if item and current:
            fields[current].append(item.group(1).strip())
            continue
        key = re.match(r"^([a-z_]+):\s*(.*)$", line)
        assert key, f"unreadable frontmatter line {line!r}"
        current, value = key.group(1), key.group(2)
        if value.startswith("["):
            fields[current] = [v.strip() for v in _top_level(value.strip()[1:-1]) if v.strip()]
        elif value:
            fields[current] = value
        else:
            fields[current] = []
    return fields


def _top_level(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


@cache
def template(kind):
    """`template` returns the frontmatter keys and section headings of *kind*'s template."""
    front, body = _split((TEMPLATES / f"{kind}.md").read_text(encoding="utf-8"))
    keys = [re.match(r"^([a-z_]+):", l).group(1) for l in front.splitlines() if re.match(r"^[a-z_]+:", l)]
    return keys, re.findall(r"^## (.+)$", body, re.M)


@cache
def _ids():
    return {_fields(_split(p.read_text(encoding="utf-8"))[0])["id"] for p in _explainers()}


@cache
def _page_text():
    """Every template, script and Python file of the application, for finding names in."""
    root = Path(settings.BASE_DIR)
    parts = []
    for pattern in ("ui/**/*.html", "assistant/**/*.html", "ui/static/**/*.js", "assistant/static/**/*.js",
                    "ui/**/*.py", "experiments/**/*.py", "core/**/*.py"):
        parts += [p.read_text(encoding="utf-8", errors="replace") for p in root.glob(pattern)]
    return "\n".join(parts)


@cache
def _url_names():
    from django.urls import get_resolver

    names = set()

    def walk(resolver, prefix=""):
        for pattern in resolver.url_patterns:
            if hasattr(pattern, "url_patterns"):
                walk(pattern, prefix + (f"{pattern.namespace}:" if pattern.namespace else ""))
            elif pattern.name:
                names.add(prefix + pattern.name)

    walk(get_resolver())
    return names


def _exists(kind, name):
    if "<" in name:
        name = name.split("<")[0]
    if kind == "figure":
        from ui.figures.catalog import FIGURES

        return name in {f.key for f in FIGURES}
    if kind == "page":
        return name in _url_names()
    if kind == "action":
        from access.policy import ACTIONS

        return name in ACTIONS
    if kind == "data-assist":
        return f'data-assist="{name}' in _page_text()
    if kind == "field" and name.startswith(("opt_", "autocompute_", "show_")):
        return True
    if name in _page_text():
        return True
    # An id a template builds from a variable, such as `{{ prefix }}-requery`.
    return "-" in name and "}}-" + name.split("-", 1)[1] + '"' in _page_text()


@pytest.mark.parametrize("path", _explainers(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_explainer_follows_its_template(path):
    """What: an explainer has its template's frontmatter fields in order, an id
    made from its kind and file name, and its template's sections in order,
    with no others. How: the file and the template are read and compared."""
    kind = FOLDERS[path.parent.name]
    keys, sections = template(kind)
    front, body = _split(path.read_text(encoding="utf-8"))
    fields = _fields(front)

    assert list(fields) == keys
    assert fields["kind"] == kind
    assert fields["id"] == f"{kind}-{path.stem.replace('_', '-')}"
    assert fields["status"] in ("draft", "reviewed")
    assert fields["title"] and fields["keywords"] and fields["about"] and fields["sources"]
    assert not re.search(r"^# ", body, re.M), "the title belongs in the frontmatter"
    assert re.findall(r"^## (.+)$", body, re.M) == sections


@pytest.mark.parametrize("path", _explainers(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_questions_are_bold_questions_with_answers(path):
    """What: "Questions people ask" holds only bold questions, each followed by
    its answer on the same line. How: each paragraph of that section is
    matched."""
    body = _split(path.read_text(encoding="utf-8"))[1]
    section = body.split("## Questions people ask", 1)[1]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    paragraphs = [p for p in paragraphs if not p.startswith("<!--")]

    assert paragraphs
    for paragraph in paragraphs:
        assert re.match(r"^\*\*[^*]+\?\*\* \S", paragraph), paragraph[:80]


@pytest.mark.parametrize("path", _explainers(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_references_point_at_what_exists(path):
    """What: every `about` target names a figure, page, field, element, page
    target or action the application has; every related explainer exists; and
    every cited file exists, with its lines inside the file. How: targets are
    looked up in the figure catalog, the URL names, the policy's actions and
    the application's templates and scripts; files on disk."""
    fields = _fields(_split(path.read_text(encoding="utf-8"))[0])
    root = Path(settings.BASE_DIR)

    for target in fields["about"]:
        kind, _, name = target.partition(":")
        assert kind in ABOUT_KINDS, f"{target}: kinds are {sorted(ABOUT_KINDS)}"
        assert _exists(kind, name), f"{target} is nowhere in the application"
    for other in fields["related"]:
        assert other in _ids() and other != fields["id"], other
    for source in fields["sources"]:
        cited = re.match(r"^([\w./-]+/[\w./-]+|[\w-]+\.(?:py|html|js|toml|md))(?::([\d, -]+))?", source)
        if cited is None or not (root / cited.group(1).split("/")[0]).is_dir() and "/" in cited.group(1):
            continue
        file = root / cited.group(1)
        assert file.exists(), f"{cited.group(1)} does not exist"
        if cited.group(2) and file.is_file():
            length = len(file.read_text(encoding="utf-8", errors="replace").splitlines())
            for line in re.findall(r"\d+", cited.group(2)):
                assert int(line) <= length, f"{source}: {file.name} has {length} lines"
