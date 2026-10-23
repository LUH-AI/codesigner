"""The tools that search the assistant's library (assistant/library.py): the
papers SMAC and Codesigner rest on, and background on tuning hyperparameters."""

from __future__ import annotations

import json

from .. import library
from .registry import Refused, ToolResult, ToolSpec


def _library():
    found = library.Library.get()
    if found is None:
        raise Refused("this instance has no library built; answer from what you know, and say "
                      "that you could not look it up")
    return found


def _shown(passage: dict, papers: dict) -> dict:
    paper = papers.get(passage["key"], {})
    out = {"id": passage["id"], "cite": f"[{passage['key']}]", "paper": paper.get("title"),
           "year": paper.get("year"), "text": passage["text"]}
    if passage.get("page"):
        out["page"] = passage["page"]
    if passage["kind"] == "summary":
        out["note"] = "our summary of the paper, not its text"
    return out


def _search_knowledge(context, query: str, k: int = 5) -> ToolResult:
    from ..llm import EmbeddingClient

    found = _library()
    hits = found.search(query, k=max(1, min(int(k), 10)), embedder=EmbeddingClient.from_settings())
    if not hits:
        return ToolResult(output="Nothing in the library matches. Say so rather than guess.")
    return ToolResult(output=json.dumps([_shown(h.passage, found.papers) for h in hits],
                                        ensure_ascii=False))


def _read_passage(context, id: int, around: int = 1) -> ToolResult:
    found = _library()
    span = max(0, min(int(around), 3))
    passages = found.around(int(id), before=span, after=span)
    if not passages:
        raise Refused(f"there is no passage {id}")
    return ToolResult(output=json.dumps([_shown(p, found.papers) for p in passages],
                                        ensure_ascii=False))


SEARCH_KNOWLEDGE = ToolSpec(
    name="search_knowledge",
    description=(
        "Search the library: the papers SMAC and Codesigner rest on (Bayesian optimization, "
        "acquisition functions, priors and their decay, Hyperband, constraints, HyperSHAP and "
        "Shapley interactions, partial dependence, ablation, projections, forecast accuracy) "
        "and background on tuning hyperparameters. Returns passages with how to cite them, "
        "e.g. [HSSL22]. Use it before explaining a method, and cite what you used."),
    parameters={"type": "object", "properties": {"query": {"type": "string"},
                                                 "k": {"type": "integer"}},
                "required": ["query"]},
    handler=_search_knowledge,
)

READ_PASSAGE = ToolSpec(
    name="read_passage",
    description="A passage the search returned, with the passages around it in the same paper.",
    parameters={"type": "object", "properties": {"id": {"type": "integer"},
                                                 "around": {"type": "integer"}},
                "required": ["id"]},
    handler=_read_passage,
)


def knowledge():
    """The library's tools."""
    return [SEARCH_KNOWLEDGE, READ_PASSAGE]
