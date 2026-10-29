"""Checking the library's sources (assistant/sources.py): a paper whose address
no longer serves it is named, with why, and under GitHub Actions annotated on
its line of the list. Against stand-in fetches, so nothing leaves the machine."""

import urllib.error

from assistant import sources

PDF = {"key": "AB20", "pdf": "https://example.org/ab.pdf"}
PAGE = {"key": "CD21", "html": "https://example.org/cd.html"}


def _serving(body=None, error=None):
    def fetch(url, timeout=60.0, first=None):
        if error is not None:
            raise error
        return body
    return fetch


def test_a_source_serving_its_paper_passes():
    """What: a PDF address that serves a PDF and a page address that serves a
    page pass, and a paper kept as its summary has nothing to check. How:
    `problem` with stand-in fetches."""
    assert sources.problem(PDF, fetch=_serving(b"%PDF-1.7 ...")) == ""
    assert sources.problem(PAGE, fetch=_serving(b"<!doctype html><p>text")) == ""
    assert sources.problem({"key": "EF99", "summary": "ours"}, fetch=_serving(error=AssertionError())) == ""


def test_a_missing_or_changed_source_says_why():
    """What: an address that answers with an error, cannot be reached, or
    serves something else than its paper fails with the reason. How: `problem`
    with stand-in fetches for each."""
    gone = urllib.error.HTTPError(PDF["pdf"], 404, "Not Found", {}, None)
    assert sources.problem(PDF, fetch=_serving(error=gone)) == f"{PDF['pdf']} answered 404 Not Found"
    down = urllib.error.URLError("Name or service not known")
    assert "could not be reached: Name or service not known" in sources.problem(PDF, fetch=_serving(error=down))
    assert sources.problem(PDF, fetch=_serving(b"<html>moved</html>")).endswith("other than a PDF")


def test_failures_are_annotated_on_their_line_under_github_actions(monkeypatch, capsys):
    """What: `main` prints each failing paper, annotates it on its line of the
    list under GitHub Actions, and exits 1. How: the real list, with every
    fetch answering 404 for one paper only."""
    papers = sources.papers()
    broken = next(p for p in papers if p.get("pdf"))

    def problem(paper, fetch=None):
        return "answered 404 Not Found" if paper["key"] == broken["key"] else ""

    monkeypatch.setattr(sources, "problem", problem)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    assert sources.main() == 1
    out = capsys.readouterr().out
    line = sources._line_of(broken["key"])
    assert line > 1
    assert "::warning file=" in out and f",line={line},title=Library source missing::" in out
    assert f"{broken['key']}: answered 404 Not Found" in out
