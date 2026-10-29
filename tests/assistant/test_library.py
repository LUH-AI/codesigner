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


# ── keeping the library current ─────────────────────────────────────────────

def test_a_fresh_build_is_current_and_names_its_list(built):
    """What: a build records the paper list's digest and its embedding model,
    and `stale` finds nothing to do afterwards. How: the fixture's build is
    read back."""
    assert built["manifest"] == library.manifest()
    assert built["embedding_model"] == "stand-in"
    assert library.stale("stand-in") == ""


def test_a_missing_library_a_changed_list_or_another_model_is_stale(built, tmp_path, monkeypatch, settings):
    """What: `stale` gives a reason when there is no index, when the paper list
    has changed since the build, and when another embedding model is
    configured. How: each condition is set up against the fixture's build."""
    assert "other-model" in library.stale("other-model")
    assert "no model" in library.stale(None)

    changed = tmp_path / "literature.toml"
    changed.write_text(library.LITERATURE.read_text() + "\n# one more line\n")
    monkeypatch.setattr(library, "LITERATURE", changed)
    assert library.stale("stand-in") == "the paper list has changed"

    settings.ASSISTANT_KNOWLEDGE_DIR = tmp_path / "elsewhere"
    assert library.stale("stand-in") == "there is no library yet"


def test_a_build_already_running_is_left_to_finish(built, settings):
    """What: a second build gives way while another holds the folder's lock.
    How: the test holds the lock itself and asks for a build."""
    with library._building(library.index_dir()) as mine:
        assert mine
        assert library.build(embedder=Embedder(), progress=lambda line: None) is None


def test_another_process_s_build_is_picked_up(built):
    """What: a library already loaded reloads once the index is rebuilt
    elsewhere, as the worker does for the web process. How: the passages and
    the build record are rewritten behind the loaded library, without
    `Library.forget`."""
    import os

    loaded = library.Library.get()
    directory = library.index_dir()
    first = loaded.passages[0]
    (directory / "passages.jsonl").write_text(json.dumps({**first, "text": "rebuilt"}) + "\n")
    (directory / "embeddings.npy").unlink()
    record = directory / "built.json"
    stamp = record.stat().st_mtime + 5
    os.utime(record, (stamp, stamp))

    again = library.Library.get()
    assert again is not loaded
    assert [p["text"] for p in again.passages] == ["rebuilt"]


def test_vectors_from_another_build_are_set_aside(built):
    """What: embeddings whose count differs from the passages', as when a
    reader catches a rebuild between files, are ignored and search goes by
    words. How: one passage is dropped from the file and the library is
    loaded fresh."""
    directory = library.index_dir()
    lines = (directory / "passages.jsonl").read_text().splitlines()
    (directory / "passages.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    library.Library.forget()

    found = library.Library.get()
    assert found.vectors is None
    assert found.search("prior decayed belief fades", k=1)


def test_ensure_builds_only_when_on_and_stale(built, settings, monkeypatch):
    """What: `assistant_knowledge ensure` does nothing while the assistant is
    off or the library is current, and queues a build when it is stale. How:
    the command runs in each state, with the queued task recorded instead of
    run."""
    from io import StringIO

    from django.core.management import call_command

    import assistant.tasks
    from assistant import llm

    queued = []
    monkeypatch.setattr(assistant.tasks, "build_library_task", lambda: queued.append(True))
    monkeypatch.setattr(llm.EmbeddingClient, "from_settings", classmethod(lambda cls: Embedder()))

    def ensure():
        out = StringIO()
        call_command("assistant_knowledge", "ensure", stdout=out)
        return out.getvalue()

    settings.ASSISTANT_ENABLED = False
    assert "off" in ensure() and not queued
    settings.ASSISTANT_ENABLED = True
    assert "current" in ensure() and not queued
    (library.index_dir() / "built.json").unlink()
    assert "queued: there is no library yet" in ensure() and queued == [True]


def test_the_queued_task_builds_a_stale_library(built, monkeypatch):
    """What: the worker's task builds the library when it is stale and leaves
    a current one alone. How: the task runs on the test queue, which runs tasks
    at once, before and after the build record is removed."""
    from assistant import llm, tasks

    monkeypatch.setattr(llm.EmbeddingClient, "from_settings", classmethod(lambda cls: Embedder()))
    before = (library.index_dir() / "built.json").read_text()

    tasks.build_library_task()
    assert (library.index_dir() / "built.json").read_text() == before

    (library.index_dir() / "built.json").unlink()
    tasks.build_library_task()
    assert library.stale("stand-in") == ""
