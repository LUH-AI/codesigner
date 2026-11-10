"""`library` searches passages of the papers in
`assistant/knowledge/literature.toml`.

`build` writes the index to `ASSISTANT_KNOWLEDGE_DIR` on each instance, and the
worker rebuilds a `stale` index on starting. `search` fuses BM25 and embedding
rankings by reciprocal rank, so it falls back to words without embeddings.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.conf import settings

from .sources import LITERATURE, fetch, link_of, papers
from .sources import source_of as _source

log = logging.getLogger(__name__)

#: Lengths in characters.
PASSAGE = 2400
OVERLAP = 300
#: Reciprocal-rank-fusion constant.
RRF = 60
#: BM25 parameters.
K1, B = 1.5, 0.75

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the
this to was were which with we our their these those can not than then also into such""".split())


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def index_dir() -> Path:
    return Path(settings.ASSISTANT_KNOWLEDGE_DIR)


def manifest() -> str:
    """`manifest` digests the paper list for `stale`."""
    return hashlib.sha256(LITERATURE.read_bytes()).hexdigest()[:16]


def built() -> dict | None:
    path = index_dir() / "built.json"
    if not (path.exists() and (index_dir() / "passages.jsonl").exists()):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def stale(embedding_model: str | None) -> str:
    """`stale` returns why a rebuild is due, or "". A None *embedding_model*
    means word search."""
    record = built()
    if record is None:
        return "there is no library yet"
    if record.get("manifest") != manifest():
        return "the paper list has changed"
    if record.get("embedding_model") != embedding_model:
        return (f"it was embedded with {record.get('embedding_model') or 'no model'}, "
                f"not {embedding_model or 'no model'}")
    return ""


@contextlib.contextmanager
def _building(directory: Path):
    """`_building` yields False while another build runs."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".building", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _replace(path: Path, write) -> None:
    """`_replace` writes atomically, so readers never see half a file."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)



def _pdf_pages(raw: bytes) -> list[str]:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return [page.extract_text() or "" for page in reader.pages]


class _Text:

    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self, html: str):
        from html.parser import HTMLParser

        parts, skipping = [], [0]
        outer = self

        class Parser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag in outer.SKIP:
                    skipping[0] += 1
                elif tag in ("p", "li", "h1", "h2", "h3", "h4", "br", "div", "tr"):
                    parts.append("\n")

            def handle_endtag(self, tag):
                if tag in outer.SKIP and skipping[0]:
                    skipping[0] -= 1

            def handle_data(self, text):
                if not skipping[0]:
                    parts.append(text)

        Parser().feed(html)
        self.text = re.sub(r"\n\s*\n+", "\n\n", "".join(parts))


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # dehyphenate
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split(pages: list[str]) -> list[tuple[int, str]]:
    """`split` returns passages cut at paragraph or sentence ends."""
    joined, starts = "", []
    for number, page in enumerate(pages, 1):
        starts.append((len(joined), number))
        joined += _clean(page) + "\n\n"
    out, position = [], 0
    while position < len(joined):
        end = min(len(joined), position + PASSAGE)
        if end < len(joined):
            window = joined[position:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "))
            if cut > PASSAGE // 2:
                end = position + cut + 1
        text = joined[position:end].strip()
        if len(text) > 80:
            page = max((n for offset, n in starts if offset <= position), default=1)
            out.append((page, text))
        if end >= len(joined):
            break
        position = max(end - OVERLAP, position + 1)
    return out


def _text_of(paper: dict) -> tuple[list[str], str]:
    """`_text_of` caches each download in the index folder for rebuilds."""
    url, form = _source(paper)
    if not url:
        return [], ""
    kept = index_dir() / "sources" / f"{re.sub(r'[^A-Za-z0-9+_-]', '_', paper['key'])}.{form}"
    if kept.exists():
        raw = kept.read_bytes()
    else:
        raw = fetch(url)
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(raw)
    if form == "pdf":
        return _pdf_pages(raw), url
    return [_Text(raw.decode("utf-8", "replace")).text], url


def build(embedder=None, only: set | None = None, progress=print) -> dict | None:
    """`build` returns None while another build runs. With *only*, it re-indexes
    the papers keyed in *only* and keeps the rest. `_text_of` reuses cached
    downloads."""
    directory = index_dir()
    with _building(directory) as mine:
        if not mine:
            progress("Another build is under way; leaving it to finish.")
            return None
        return _build(directory, embedder, only, progress)


def _build(directory: Path, embedder, only, progress) -> dict:
    kept = {}
    if only and (directory / "passages.jsonl").exists():
        for passage in _read_passages(directory):
            if passage["key"] not in only:
                kept.setdefault(passage["key"], []).append(passage)
    passages, failed = [], {}
    for paper in papers():
        key = paper["key"]
        if only and key not in only:
            passages += kept.get(key, [])
            continue
        found = []
        if paper.get("arxiv") or paper.get("pdf") or paper.get("html"):
            try:
                pages, source = _text_of(paper)
                found = [{"key": key, "page": page, "text": text, "kind": "paper", "source": source}
                         for page, text in split(pages)]
                time.sleep(1.0 if paper.get("arxiv") else 0.2)
            except Exception as error:  # noqa: BLE001 — one paper missing, not the library
                failed[key] = str(error)[:200]
        if not found and paper.get("summary"):
            found = [{"key": key, "page": None, "text": paper["summary"], "kind": "summary",
                      "source": ""}]
        if not found:
            found = [{"key": key, "page": None, "kind": "summary", "source": "",
                      "text": f"{paper['title']} ({paper['authors']}, {paper['year']})."}]
        passages += found
        progress(f"{key}: {len(found)} passages" + (f" (failed: {failed[key]})" if key in failed else ""))
        if key in failed:
            log.warning("The library could not fetch %s from %s: %s. It is indexed by its "
                        "summary instead.", key, _source(paper)[0], failed[key])
    for number, passage in enumerate(passages):
        passage["id"] = number
    vectors = None
    if embedder is not None:
        vectors = embedder.embed([f"{_title(p['key'])}\n{p['text']}" for p in passages])

    def write_passages(path):
        with open(path, "w", encoding="utf-8") as out:
            for passage in passages:
                out.write(json.dumps(passage, ensure_ascii=False) + "\n")

    _replace(directory / "passages.jsonl", write_passages)
    if vectors is not None:
        def write_vectors(path):
            with open(path, "wb") as out:
                np.save(out, vectors)

        _replace(directory / "embeddings.npy", write_vectors)
    elif (directory / "embeddings.npy").exists():
        (directory / "embeddings.npy").unlink()
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "manifest": manifest(),
              "passages": len(passages), "papers": len(papers()), "failed": failed,
              "embedding_model": getattr(embedder, "model", None) if vectors is not None else None}
    _replace(directory / "built.json",
             lambda path: path.write_text(json.dumps(record, indent=2), encoding="utf-8"))
    Library.forget()
    return record


def _title(key: str) -> str:
    return next((p["title"] for p in papers() if p["key"] == key), key)


def _read_passages(directory: Path) -> list[dict]:
    with open(directory / "passages.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]



@dataclass
class Hit:
    passage: dict
    score: float


class Library:
    """`Library` caches the index."""

    _loaded: "Library | None" = None
    _lock = threading.Lock()

    def __init__(self, directory: Path):
        self.passages = _read_passages(directory)
        self.papers = {p["key"]: p for p in papers()}
        path = directory / "embeddings.npy"
        self.vectors = np.load(path) if path.exists() else None
        # Mismatched vectors mean a build is running.
        if self.vectors is not None and len(self.vectors) != len(self.passages):
            self.vectors = None
        built = directory / "built.json"
        self.embedding_model = (json.loads(built.read_text()).get("embedding_model")
                                if built.exists() else None)
        self._terms = [Counter(words(f"{self.papers.get(p['key'], {}).get('title', '')} {p['text']}"))
                       for p in self.passages]
        self._lengths = np.array([sum(t.values()) for t in self._terms], dtype=float)
        self._average = float(self._lengths.mean()) if len(self._lengths) else 1.0
        frequency = Counter(term for terms in self._terms for term in terms)
        n = len(self._terms)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    #: `get` reloads when `built.json` changes.
    _stamp: float | None = None

    @classmethod
    def get(cls) -> "Library | None":
        record = index_dir() / "built.json"
        stamp = record.stat().st_mtime if record.exists() else None
        with cls._lock:
            if cls._loaded is not None and stamp != cls._stamp:
                cls._loaded = None
            if cls._loaded is None and (index_dir() / "passages.jsonl").exists():
                cls._loaded = cls(index_dir())
                cls._stamp = stamp
            return cls._loaded

    @classmethod
    def forget(cls) -> None:
        with cls._lock:
            cls._loaded = None

    def bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.passages))
        for term in set(words(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([terms.get(term, 0) for terms in self._terms], dtype=float)
            scores += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self._lengths / self._average))
        return scores

    def search(self, query: str, k: int = 5, kind: str = "all", embedder=None) -> list[Hit]:
        candidates = np.arange(len(self.passages))
        if kind != "all":
            candidates = np.array([i for i, p in enumerate(self.passages)
                                   if (p["kind"] == "codesigner") == (kind == "codesigner")], dtype=int)
        if not len(candidates):
            return []
        rankings = []
        lexical = self.bm25(query)[candidates]
        if lexical.any():
            rankings.append(candidates[np.argsort(-lexical)][:200])
        if self.vectors is not None and embedder is not None:
            try:
                prefix = settings.ASSISTANT_EMBED_QUERY_PREFIX
                semantic = self.vectors[candidates] @ embedder.embed([prefix + query])[0]
                rankings.append(candidates[np.argsort(-semantic)][:200])
            except Exception:  # noqa: BLE001 — words alone, when the embedder is away
                log.info("searching the library by words alone", exc_info=True)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (RRF + rank + 1)
        hits, per_paper = [], Counter()
        for index, score in sorted(fused.items(), key=lambda item: -item[1]):
            passage = self.passages[index]
            if per_paper[passage["key"]] >= 2:
                continue
            per_paper[passage["key"]] += 1
            hits.append(Hit(passage, score))
            if len(hits) >= k:
                break
        return hits

    def around(self, passage_id: int, before: int = 1, after: int = 1) -> list[dict]:
        if not 0 <= passage_id < len(self.passages):
            return []
        key = self.passages[passage_id]["key"]
        span = range(max(0, passage_id - before), min(len(self.passages), passage_id + after + 1))
        return [self.passages[i] for i in span if self.passages[i]["key"] == key]


#: For example [LEF+22, HHL11].
_KEY = r"[A-Za-z]+\+?\d{2}"
CITATION = re.compile(rf"\[({_KEY}(?:[,;] *{_KEY})*)\]")


def _cited_keys(text: str) -> list[str]:
    out = []
    for match in CITATION.finditer(text or ""):
        out += [key for key in re.split(r"[,;] *", match.group(1)) if key not in out]
    return out


def uncited_keys(text: str) -> list[str]:
    """`uncited_keys` returns cited keys missing from the library."""
    keys = {p["key"] for p in papers()}
    return [key for key in _cited_keys(text) if key not in keys]


def cited(text: str) -> list[dict]:
    keys = {p["key"]: p for p in papers()}
    return [{"key": key, "title": keys[key]["title"], "authors": keys[key]["authors"],
             "year": keys[key]["year"], "link": link_of(keys[key])}
            for key in _cited_keys(text) if key in keys]
