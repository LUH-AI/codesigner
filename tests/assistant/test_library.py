"""The assistant's library (assistant/library.py): papers split into passages,
searched by words and by meaning, papers without an open source kept as their
summary, the search still working without embeddings, and the papers an answer
cites listed under it.

Against a small list and stand-in fetches, so nothing leaves the machine.
"""

import json

import numpy as np
import pytest

from assistant import library
from assistant.gateway import Context
from assistant.models import AssistantEvent, Conversation
from assistant.turns import tools_for

pytestmark = pytest.mark.django_db

PAPERS = [
    {"key": "AB20", "title": "Priors in Bayesian Optimization", "authors": "A, B", "year": 2020,
     "arxiv": "0000.00001"},
    {"key": "CD21", "title": "Forecast Accuracy", "authors": "C, D", "year": 2021,
     "pdf": "https://example.org/cd.pdf"},
    {"key": "EF99", "title": "Expected Improvement", "authors": "E, F", "year": 1999,
     "summary": "Choose the next point by its expected improvement over the best so far."},
]

PAGES = {
    "AB20": ["A prior over the optimum is decayed as trials accumulate, so the belief fades "
             "and the data takes over. " * 30, "Unrelated discussion of kernels. " * 200],
    "CD21": ["The mean absolute scaled error divides errors by those of a naive forecast. " * 30],
}


class Embedder:
    """Vectors from a handful of words, so meaning can be stood in for."""

    model = "stand-in"
    AXES = ("prior", "forecast", "improvement")

    def embed(self, texts):
        rows = np.array([[text.lower().count(w) + 0.01 for w in self.AXES] for text in texts], float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture
def built(tmp_path, settings, monkeypatch):
    settings.ASSISTANT_KNOWLEDGE_DIR = tmp_path / "library"
    monkeypatch.setattr(library, "papers", lambda: PAPERS)
    monkeypatch.setattr(library, "_text_of", lambda paper: (PAGES[paper["key"]], "stand-in"))
    monkeypatch.setattr(library.time, "sleep", lambda seconds: None)
    library.Library.forget()
    report = library.build(embedder=Embedder(), progress=lambda line: None)
    yield report
    library.Library.forget()


def test_papers_become_passages_and_summaries_stand_in_for_the_rest(built):
    """What: fetched papers are split into overlapping passages that remember
    their page; a paper with no open source is one passage, its summary,
    marked as ours.
    How: a build of the three-paper list."""
    found = library.Library.get()
    by_key = {}
    for passage in found.passages:
        by_key.setdefault(passage["key"], []).append(passage)

    assert len(by_key["AB20"]) > 1 and {p["page"] for p in by_key["AB20"]} == {1, 2}
    assert all(len(p["text"]) <= library.PASSAGE for p in found.passages)
    assert [p["kind"] for p in by_key["EF99"]] == ["summary"]
    assert built["passages"] == len(found.passages) and found.vectors.shape[0] == len(found.passages)


def test_the_search_finds_the_passage_that_answers(built):
    """What: a question about decaying priors finds the priors paper, one
    about scaled errors the forecasting paper, by words and by meaning.
    How: two searches with the stand-in embedder."""
    found = library.Library.get()

    assert found.search("how is the prior decayed", k=1, embedder=Embedder())[0].passage["key"] == "AB20"
    assert found.search("scaled error of a forecast", k=1, embedder=Embedder())[0].passage["key"] == "CD21"


def test_without_embeddings_the_search_goes_by_words(built):
    """What: with the embedding server away, or none configured, the search
    still answers, by words alone.
    How: an embedder that fails, and none at all."""
    class Away:
        def embed(self, texts):
            raise ConnectionError("down")

    found = library.Library.get()

    assert found.search("naive forecast scaled error", k=1, embedder=Away())[0].passage["key"] == "CD21"
    assert found.search("naive forecast scaled error", k=1)[0].passage["key"] == "CD21"


def test_a_search_gives_at_most_two_passages_of_one_paper(built):
    """What: one long paper cannot fill every slot of an answer.
    How: a search all of AB20's passages match."""
    hits = library.Library.get().search("prior decayed belief fades", k=5)

    assert sum(h.passage["key"] == "AB20" for h in hits) <= 2


def test_rebuilding_one_paper_keeps_the_others(built, monkeypatch):
    """What: `only` fetches just the papers named and keeps the rest.
    How: a rebuild of CD21 with a fetch that would fail for anything else."""
    def only_cd(paper):
        assert paper["key"] == "CD21"
        return PAGES["CD21"], "stand-in"

    monkeypatch.setattr(library, "_text_of", only_cd)
    library.build(embedder=None, only={"CD21"}, progress=lambda line: None)

    assert {p["key"] for p in library.Library.get().passages} == {"AB20", "CD21", "EF99"}


def test_an_answer_lists_the_papers_it_cites(built):
    """What: keys cited as [KEY] in an answer are listed under it, once each,
    with their link; text that only looks like a key is not.
    How: `cited` on an answer citing AB20 twice and a made-up key."""
    found = library.cited("Priors fade [AB20], as shown [AB20]; see also [ZZ99].")

    assert [p["key"] for p in found] == ["AB20"]
    assert found[0]["link"] == "https://arxiv.org/abs/0000.00001"


def test_the_tools_search_and_read_around(built):
    """What: search_knowledge returns passages with how to cite them;
    read_passage returns one with its neighbours from the same paper.
    How: the tools on a conversation, against the built library."""
    conversation = Conversation.objects.create()
    tools = tools_for(conversation)
    hits = json.loads(tools.execute("search_knowledge", {"query": "summary expected improvement"},
                                    Context(conversation)).output)
    summary = next(h for h in hits if h["cite"] == "[EF99]")
    first = next(h for h in hits if h["cite"] == "[AB20]") if any(h["cite"] == "[AB20]" for h in hits) \
        else hits[0]
    around = json.loads(tools.execute("read_passage", {"id": first["id"]}, Context(conversation)).output)

    assert summary["note"] == "our summary of the paper, not its text"
    assert all(p["cite"] == first["cite"] for p in around)


def test_without_a_library_the_tool_says_so(settings, tmp_path):
    """What: an instance with no library built tells the model to say it could
    not look it up, instead of failing.
    How: search_knowledge with an empty index folder."""
    settings.ASSISTANT_KNOWLEDGE_DIR = tmp_path / "none"
    library.Library.forget()
    conversation = Conversation.objects.create()
    result = tools_for(conversation).execute("search_knowledge", {"query": "priors"},
                                             Context(conversation))

    assert "no library built" in result.output


def test_a_cited_answer_puts_its_sources_under_the_reply(built, monkeypatch):
    """What: when the model's answer cites a paper, a sources event follows it.
    How: a scripted answer citing AB20."""
    from assistant.loop import run_turn

    from .helpers import MockLLM, text

    conversation = Conversation.objects.create()
    run_turn(conversation, "how do priors fade?", llm=MockLLM(text("They decay [AB20].")),
             tools=tools_for(conversation), stream=False)
    sources = AssistantEvent.objects.get(conversation=conversation, kind="sources")

    assert [p["key"] for p in sources.payload["papers"]] == ["AB20"]
