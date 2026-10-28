"""`pages` builds the file list, pages and search hits."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path

from . import source as S

ROOT = Path(__file__).resolve().parents[2]

SKIP = ("migrations",)

try:
    from pygments.lexers import PythonLexer
    from pygments.token import STANDARD_TYPES
except ImportError:
    PythonLexer = None


def files() -> list[str]:
    """`files` lists tracked or addable Python files."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        out = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    return sorted(f for f in out
                  if f and not any(part in SKIP for part in f.split("/"))
                  and (ROOT / f).is_file())


def module_name(path: str) -> str:
    parts = path[:-3].split("/")
    if parts[0] == "model_sdk":  # installed as codesigner_model
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or path


_loaded: dict[str, tuple[int, str, S.Parsed | None, str | None]] = {}


def load(path: str):
    """`load` caches `(mtime, source, parsed, error)` per path."""
    mtime = (ROOT / path).stat().st_mtime_ns
    hit = _loaded.get(path)
    if hit and hit[0] == mtime:
        return hit
    source = (ROOT / path).read_text(encoding="utf-8")
    try:
        entry = (mtime, source, S.parse(source), None)
    except (SyntaxError, S.EditError) as e:
        entry = (mtime, source, None, str(e))
    _loaded[path] = entry
    return entry


def mtime(path: str) -> int:
    return (ROOT / path).stat().st_mtime_ns


def index():
    out, names = [], {}
    for path in files():
        _, _, p, broken = load(path)
        mod = module_name(path)
        names[mod] = path
        names[path] = path
        entry = {"path": path, "module": mod, "broken": broken}
        if p:
            doc = p.module_doc.text.strip().split("\n")[0] if p.module_doc else ""
            entry.update(summary=doc, symbols=len(p.symbols),
                         documented=sum(1 for s in p.symbols if s.doc))
            for s in p.symbols:
                names.setdefault(f"{mod}.{s.qualname}", f"{path}:{s.qualname}")
        out.append(entry)
    return {"files": out, "names": names, "width": S.WIDTH, "root": str(ROOT)}


def _highlight(source: str) -> list[str]:
    """`_highlight` returns HTML lines, each closing its spans."""
    if PythonLexer is None:
        return [html.escape(line) for line in source.split("\n")]
    lines = [[]]
    for ttype, value in PythonLexer(stripnl=False, ensurenl=False).get_tokens(source):
        while ttype not in STANDARD_TYPES:
            ttype = ttype.parent
        cls = STANDARD_TYPES[ttype]
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                text = html.escape(part)
                lines[-1].append(f'<span class="{cls}">{text}</span>' if cls else text)
    return ["".join(parts) for parts in lines]


def page(path: str):
    mt, source, p, broken = load(path)
    base = {"path": path, "module": module_name(path), "mtime": mt}
    if p is None:
        return base | {"broken": broken, "segments": [
            {"type": "code", "lines": [{"n": i + 1, "html": html.escape(line), "role": "body"}
                                       for i, line in enumerate(source.split("\n"))]}]}
    covered = {line: r for r in p.regions for line in range(r.start, r.end + 1)}
    headers = {}
    for s in p.symbols:
        for line in range(s.start, s.header_end + 1):
            headers.setdefault(line, s)
    targets = {r.end + 1 for r in p.regions if r.kind == "attr"}
    lit = _highlight(source)
    segments, n, i = [], len(p.lines), 1
    while i <= n:
        r = covered.get(i)
        if r:
            segments.append({"type": "prose"} | r.as_json())
            i = r.end + 1
            continue
        if not segments or segments[-1]["type"] != "code":
            segments.append({"type": "code", "lines": []})
        line = {"n": i, "html": lit[i - 1]}
        s = headers.get(i)
        if s:
            line["role"] = "header"
            if i == s.start:
                line |= {"sym": s.qualname, "kind": s.kind, "depth": s.qualname.count(".")}
            if i == s.header_end and s.can_add:
                line["add"] = {"qualname": s.qualname, "start": s.start,
                               "indent": s.body_indent, "width": S.WIDTH - s.body_indent}
        elif i in targets:
            line["role"] = "target"
        elif not p.lines[i - 1].strip():
            line["role"] = "blank"
        else:
            line["role"] = "body"
        segments[-1]["lines"].append(line)
        i += 1
    return base | {"segments": segments,
                   "symbols": [s.qualname for s in p.symbols]}


def search(query: str, limit: int = 150):
    words = query.lower().split()
    if not words:
        return []
    hits = []
    for path in files():
        _, _, p, _ = load(path)
        if not p:
            continue
        for s in p.symbols:
            if all(w in s.qualname.lower() for w in words):
                hits.append({"path": path, "line": s.start, "anchor": s.qualname,
                             "kind": "symbol", "title": f"{s.kind} {s.qualname}",
                             "snippet": ""})
        for r in p.regions:
            low = r.text.lower()
            if not all(w in low for w in words):
                continue
            at = low.index(words[0])
            a, b = max(0, at - 70), at + 130
            flat = " ".join(r.text[a:b].split())
            hits.append({"path": path, "line": r.start, "anchor": f"L{r.start}",
                         "kind": r.kind,
                         "title": r.owner if r.owner else ("module" if r.owner == "" else r.kind),
                         "snippet": ("…" if a else "") + flat + ("…" if b < len(r.text) else "")})
        if len(hits) >= limit:
            break
    hits.sort(key=lambda h: h["kind"] != "symbol")
    return hits[:limit]
