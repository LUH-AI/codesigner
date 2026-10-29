"""Where the assistant's library gets its papers: the list in
assistant/knowledge/literature.toml, each paper's open-access address, and a
check that every address still serves the paper.

Only the standard library, so the check runs without Codesigner installed:

    python -m assistant.sources

prints each paper whose source fails and exits 1 when any does. Under GitHub
Actions each failure is also an annotation on its line of literature.toml
(.github/workflows/library-sources.yml runs it on every push).
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

LITERATURE = Path(__file__).resolve().parent / "knowledge" / "literature.toml"
AGENT = "Codesigner library builder"


def papers() -> list[dict]:
    """`papers` returns the library's list."""
    with open(LITERATURE, "rb") as handle:
        return tomllib.load(handle)["paper"]


def source_of(paper: dict) -> tuple[str, str]:
    """`source_of` returns where *paper*'s text is and its form, "pdf" or
    "html", or two empty strings for a paper kept as its summary."""
    if paper.get("arxiv"):
        return f"https://arxiv.org/pdf/{paper['arxiv']}", "pdf"
    if paper.get("pdf"):
        return paper["pdf"], "pdf"
    if paper.get("html"):
        return paper["html"], "html"
    return "", ""


def link_of(paper: dict) -> str:
    """`link_of` returns the page a person reading the citation is sent to."""
    if paper.get("arxiv"):
        return f"https://arxiv.org/abs/{paper['arxiv']}"
    return paper.get("pdf") or paper.get("html") or ""


def fetch(url: str, timeout: float = 60.0, first: int | None = None) -> bytes:
    """`fetch` returns the body at *url*, or only its first *first* bytes when
    the server honours a range."""
    headers = {"User-Agent": AGENT}
    if first:
        headers["Range"] = f"bytes=0-{first - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return response.read(first) if first else response.read()


def problem(paper: dict, fetch=fetch) -> str:
    """`problem` returns why *paper*'s source fails to serve it, or "" when it
    serves a PDF or a page with text as listed."""
    url, form = source_of(paper)
    if not url:
        return ""
    try:
        head = fetch(url, timeout=30, first=4096)
    except urllib.error.HTTPError as error:
        return f"{url} answered {error.code} {error.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return f"{url} could not be reached: {getattr(error, 'reason', error)}"
    if form == "pdf" and not head.lstrip().startswith(b"%PDF"):
        return f"{url} served something other than a PDF"
    if form == "html" and b"<" not in head:
        return f"{url} served no web page"
    return ""


def _line_of(key: str) -> int:
    for number, line in enumerate(LITERATURE.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() == f'key = "{key}"':
            return number
    return 1


def main() -> int:
    """`main` checks every paper's source and reports each that fails."""
    found = []
    for paper in papers():
        reason = problem(paper)
        if reason:
            found.append((paper["key"], reason))
        if paper.get("arxiv"):
            time.sleep(1.0)
    where = LITERATURE.relative_to(Path.cwd()) if LITERATURE.is_relative_to(Path.cwd()) else LITERATURE
    for key, reason in found:
        print(f"{key}: {reason}")
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::warning file={where},line={_line_of(key)},title=Library source missing::"
                  f"{key}: {reason}")
    checked = sum(1 for p in papers() if source_of(p)[0])
    print(f"{checked - len(found)} of {checked} sources serve their paper.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
