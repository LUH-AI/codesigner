"""A `Region` is a docstring or a whole-line `#`/`#:` block, and an edit
replaces only its lines. `check` refuses edits that change the docstring-free
syntax tree or add a SyntaxWarning.
"""

from __future__ import annotations

import ast
import bisect
import functools
import io
import re
import textwrap
import tokenize
import warnings
from dataclasses import dataclass, field

#: Wrap column, counting indent and markers.
WIDTH = 80

# Tool-addressed comments, which stay code.
PRAGMA = re.compile(r"#\s*(?:!|-\*-|noqa|type:|pragma|fmt:|pylint:|pyright:|mypy:|isort:)")

LIST_ITEM = re.compile(r"(?:[-*+•]|\d+[.)])\s+")

# Hand-laid text, kept unwrapped.
ALIGNED = re.compile(r"\S {3,}\S|[^\s.!?:] {2}\S")
LITERAL_START = ("```", ">>>", "|", "+-", "===", "---")

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class EditError(Exception):
    """`EditError` carries a refused edit's reason."""


class Conflict(EditError):
    """The prose moved since the page loaded."""


@dataclass
class Region:
    kind: str           # "docstring", "comment" or "attr"
    start: int          # 1-based
    end: int            # inclusive
    indent: int         # prose column
    text: str           # dedented, unmarked
    raw: str            # verbatim, for conflicts
    owner: str | None = None   # qualname; "" is the module
    marker: str = ""
    opening: str = ""   # e.g. 'r"""'
    quote: str = ""
    lead: bool = False  # text starts after the opening line
    trail: bool = False  # closing quote on its own line
    head: str = ""      # text before the literal
    tail: str = ""      # text after it, newline included

    @property
    def first_offset(self):
        """`first_offset` counts the opening quote's columns."""
        return 0 if self.kind != "docstring" or self.lead else len(self.opening)

    @property
    def last_offset(self):
        return 0 if self.kind != "docstring" or self.trail else len(self.quote)

    @property
    def width(self):
        """`width` gives usable columns per line."""
        if self.kind == "docstring":
            return WIDTH - self.indent
        return WIDTH - self.indent - len(self.marker) - 1

    def as_json(self):
        return {
            "kind": self.kind, "start": self.start, "end": self.end,
            "indent": self.indent, "text": self.text, "flow": flow(self.text),
            "raw": self.raw, "owner": self.owner, "width": self.width,
            "first": self.first_offset, "last": self.last_offset,
        }


@dataclass
class Symbol:
    qualname: str
    kind: str
    start: int
    header_end: int     # signature's closing colon
    body_start: int
    body_indent: int
    doc: Region | None = None

    @property
    def can_add(self):
        """`can_add` holds when a docstring can sit below the header."""
        return self.doc is None and self.body_start > self.header_end


@dataclass
class Parsed:
    source: str
    lines: list[str]
    regions: list[Region] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    module_doc: Region | None = None


# reading


@functools.lru_cache(maxsize=16)
def parse(source: str) -> Parsed:
    if "\r" in source:
        raise EditError("this file has CRLF line endings, which this tool does not write")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    p = Parsed(source, source.splitlines(keepends=True))
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    starts = [t.start for t in tokens]

    def visit(node, qual):
        doc = _docstring(p, node, qual)
        if doc:
            p.regions.append(doc)
        if isinstance(node, ast.Module):
            p.module_doc = doc
        else:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            p.symbols.append(Symbol(
                qual, "class" if isinstance(node, ast.ClassDef)
                else "async def" if isinstance(node, ast.AsyncFunctionDef) else "def",
                first, _header_end(tokens, starts, node), node.body[0].lineno,
                _col(p.lines[node.body[0].lineno - 1], node.body[0].col_offset), doc))
        for child in _children(node):
            visit(child, f"{qual}.{child.name}" if qual else child.name)

    visit(tree, "")
    p.regions.extend(_comment_blocks(p, tokens, {r.start for r in p.regions}))
    p.regions.sort(key=lambda r: r.start)
    p.symbols.sort(key=lambda s: s.start)
    return p


def _children(node):
    """`_children` also yields defs and classes inside if or try blocks."""
    stack = list(reversed(node.body))
    while stack:
        n = stack.pop()
        if isinstance(n, DEFS):
            yield n
        else:
            for name in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(reversed(getattr(n, name, [])))


def _col(line: str, byte_col: int) -> int:
    """`_col` maps UTF-8 byte offsets to character columns."""
    return len(line.encode()[:byte_col].decode())


def _is_doc(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _docstring(p: Parsed, node, qual) -> Region | None:
    if not node.body or not _is_doc(node.body[0]):
        return None
    e = node.body[0]
    first, last = p.lines[e.lineno - 1], p.lines[e.end_lineno - 1]
    c0, c1 = _col(first, e.col_offset), _col(last, e.end_col_offset)
    head, tail = first[:c0], last[c1:]
    # A docstring sharing its def line stays code.
    if head.strip() or tail.strip():
        return None
    src = "".join(p.lines[e.lineno - 1:e.end_lineno])
    literal = src[c0:len(src) - len(tail)]
    m = re.match(r"([rRuU]?)(\"\"\"|'''|\"|')", literal)
    if not m or not literal.endswith(m.group(2)) or not _one_string(literal):
        return None
    opening, quote = m.group(0), m.group(2)
    inner = literal[len(opening):-len(quote)]
    lead = inner.startswith("\n")
    if lead:
        inner = inner[1:]
    t = re.search(r"\n[ \t]*\Z", inner)
    trail = bool(t)
    if trail:
        inner = inner[:t.start()]
    pad = " " * c0
    out = []
    for i, line in enumerate(inner.split("\n")):
        if i == 0 and not lead:
            out.append(line)
        else:
            out.append(line[c0:] if line.startswith(pad) else line.lstrip())
    return Region("docstring", e.lineno, e.end_lineno, c0, "\n".join(out),
                  "".join(p.lines[e.lineno - 1:e.end_lineno]), owner=qual,
                  opening=opening, quote=quote, lead=lead, trail=trail,
                  head=head, tail=tail)


def _one_string(literal):
    """`_one_string` rejects implicit concatenation."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(literal).readline)
                if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)]
    except tokenize.TokenError:
        return False
    return len(toks) == 1 and toks[0].type == tokenize.STRING


def _header_end(tokens, starts, node):
    """`_header_end` locates the signature's closing colon."""
    i = bisect.bisect_left(starts, (node.lineno, 0))
    depth = 0
    for t in tokens[i:]:
        if t.type == tokenize.OP:
            if t.string in "([{":
                depth += 1
            elif t.string in ")]}":
                depth -= 1
            elif t.string == ":" and depth == 0:
                return t.start[0]
    return node.lineno


def _comment_blocks(p: Parsed, tokens, taken):
    run = []

    def close():
        if run:
            line0, col, marker = run[0][0], run[0][1], run[0][2]
            n = len(marker)
            text = "\n".join(
                s[n + 1:] if s[n:n + 1] == " " else s[n:] for _, _, _, s in run)
            regions.append(Region(
                "attr" if marker == "#:" else "comment", line0, run[-1][0], col,
                text, "".join(p.lines[line0 - 1:run[-1][0]]), marker=marker,
                head=p.lines[line0 - 1][:col]))
            run.clear()

    regions = []
    for t in tokens:
        if t.type != tokenize.COMMENT:
            if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                close()
            continue
        line, col = t.start
        whole = not p.lines[line - 1][:col].strip()
        if not whole or PRAGMA.match(t.string) or line in taken:
            close()
            continue
        marker = "#:" if t.string.startswith("#:") else "#"
        if run and (run[-1][0] != line - 1 or run[-1][1] != col or run[-1][2] != marker):
            close()
        run.append((line, col, marker, t.string))
    close()
    return regions


# wrapping


@dataclass
class Unit:
    kind: str
    lines: list[str]

    def key(self):
        if self.kind == "literal":
            return "\n".join(self.lines)
        return " ".join(line.strip() for line in self.lines)


def _blocks(text):
    """`_blocks` splits *text* at blank lines."""
    out, cur, gap = [], [], 0
    for line in text.split("\n"):
        if line.strip():
            cur.append(line)
            continue
        if cur:
            out.append((gap, cur))
            cur, gap = [], 0
        gap += 1
    if cur:
        out.append((gap, cur))
        gap = 0
    return out, gap


def _units(block):
    units = []
    for line in block:
        if LIST_ITEM.match(line):
            units.append(Unit("item", [line]))
        elif line[:1].isspace() and units and units[-1].kind == "item":
            units[-1].lines.append(line)
        elif not line[:1].isspace() and units and units[-1].kind == "para":
            units[-1].lines.append(line)
        else:
            units.append(Unit("para", [line]))
    literal = any(
        u.kind == "para" and any(l[:1].isspace() for l in u.lines) for u in units
    ) or any(l.lstrip().startswith(LITERAL_START) or ALIGNED.search(l.strip()) for l in block)
    return [Unit("literal", block)] if literal else units


def flow(text: str) -> str:
    """Paragraphs and items joined onto single lines."""
    blocks, gap = _blocks(text)
    parts = []
    for before, block in blocks:
        units = _units(block)
        parts.append("\n" * before + "\n".join(u.key() for u in units))
    return "\n".join(parts) + "\n" * gap


def reflow(old: str, new: str, width: int, first: int = 0, last: int = 0) -> str:
    """Unchanged units keep their old lines, and the rest wrap to *width*."""
    kept, was_last = {}, None
    for _, block in _blocks(old)[0]:
        for u in _units(block):
            kept.setdefault(u.key(), u.lines)
            was_last = u.key()
    blocks, gap = _blocks(new)
    out, last_unit = [], None
    for before, block in blocks:
        out.extend([""] * before)
        for u in _units(block):
            if u.key() in kept:
                lines = kept[u.key()]
            elif u.kind == "literal":
                lines = u.lines
            else:
                lines = _wrap(u, width, width - first if not out else width)
            out.extend(lines)
            last_unit = u
    # Only a changed last line needs room for the quote.
    if (last and out and last_unit.kind != "literal" and last_unit.key() != was_last
            and len(out[-1]) > width - last):
        tail = out.pop()
        indent = len(tail) - len(tail.lstrip())
        out.extend(textwrap.wrap(tail, width - last, subsequent_indent=" " * indent,
                                 break_long_words=False, break_on_hyphens=False))
    return "\n".join(out + [""] * gap) if gap else "\n".join(out)


def _wrap(u: Unit, width: int, first_width: int):
    text = u.key()
    hang = len(LIST_ITEM.match(text).group(0)) if u.kind == "item" else 0
    opts = dict(break_long_words=False, break_on_hyphens=False)
    head = textwrap.wrap(text, first_width, **opts)[:1]
    if not head:
        return [text]
    rest = text[len(head[0]):].strip()
    return head + (textwrap.wrap(rest, width, initial_indent=" " * hang,
                                 subsequent_indent=" " * hang, **opts) if rest else [])


# writing


def region_lines(r: Region, text: str) -> list[str]:
    """`region_lines` renders nonblank *text* as source lines."""
    if not text.strip():
        return []
    if r.kind == "docstring":
        quote = r.quote
        if "\n" in text and len(quote) == 1:
            quote = quote * 3
        opening = r.opening[:-len(r.quote)] + quote
        pad = " " * r.indent
        body = [line if i == 0 and not r.lead else (pad + line if line else "")
                for i, line in enumerate(text.split("\n"))]
        literal = ("\n" if r.lead else "") + "\n".join(body) + ("\n" + pad if r.trail else "")
        return (r.head + opening + literal + quote + r.tail).splitlines(keepends=True)
    old_text = r.text.split("\n")
    old_src = r.raw.splitlines(keepends=True)
    out = []
    for i, line in enumerate(text.split("\n")):
        if i < len(old_text) and line == old_text[i]:
            out.append(old_src[i] if old_src[i].endswith("\n") else old_src[i] + "\n")
        else:
            out.append(r.head + r.marker + (" " + line if line else "") + "\n")
    if not r.raw.endswith("\n"):
        out[-1] = out[-1][:-1]
    return out


@functools.lru_cache(maxsize=16)
def _shape(source):
    """`_shape` dumps the docstring-free syntax tree of *source*."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, *DEFS)) and node.body and _is_doc(node.body[0]):
            node.body = node.body[1:]
    return ast.dump(tree)


@functools.lru_cache(maxsize=16)
def _syntax_warnings(source):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, "<docsite>", "exec")
    return sorted(str(w.message) for w in caught)


def check(old: str, new: str):
    """`check` refuses changes outside the prose."""
    if new == old:
        return
    try:
        new_shape = _shape(new)
    except SyntaxError as e:
        raise EditError(f"that would not parse: {e.msg} (line {e.lineno})") from None
    if new_shape != _shape(old):
        raise EditError("that would change the code, not just the prose — "
                        "a docstring ended early, or a statement became one")
    added = set(_syntax_warnings(new)) - set(_syntax_warnings(old))
    if added:
        raise EditError(f"{min(added)}: write it as \\\\ or make the docstring r\"\"\"")


def locate(p: Parsed, start: int, raw: str) -> Region:
    """`locate` finds the region by position or by unique text."""
    for r in p.regions:
        if r.start == start and r.raw == raw:
            return r
    found = [r for r in p.regions if r.raw == raw]
    if len(found) == 1:
        return found[0]
    raise Conflict("the file changed under this text — reload to see it as it is now")


def replace_region(source: str, start: int, raw: str, text: str, flowed: bool):
    """`replace_region` writes *text* into the region, returning source and
    line."""
    p = parse(source)
    r = locate(p, start, raw)
    if flowed:
        text = reflow(r.text, text, r.width, r.first_offset, r.last_offset)
    text = text.strip("\n")
    if r.kind == "docstring" and r.quote in text:
        raise EditError(f"the text contains {r.quote}, which would end the docstring")
    new = "".join(p.lines[:r.start - 1] + region_lines(r, text) + p.lines[r.end:])
    check(source, new)
    return new, r.start


def add_docstring(source: str, qualname: str, near: int, text: str, flowed: bool):
    """`add_docstring` returns *source* with *text* as *qualname*'s new
    docstring."""
    p = parse(source)
    cands = [s for s in p.symbols if s.qualname == qualname]
    if not cands:
        raise Conflict(f"{qualname} is no longer in this file — reload")
    s = min(cands, key=lambda s: abs(s.start - near))
    if not s.can_add:
        raise EditError(f"{qualname} already has a docstring, or its body shares its def line")
    r = Region("docstring", s.header_end + 1, s.header_end, s.body_indent, "", "",
               opening='"""', quote='"""', head=" " * s.body_indent, tail="\n")
    text = text.strip("\n")
    if flowed:
        text = reflow("", text, r.width, r.first_offset, r.last_offset)
    if not text.strip():
        raise EditError("nothing to add")
    if '"""' in text:
        raise EditError('the text contains """, which would end the docstring')
    new = "".join(p.lines[:s.header_end] + region_lines(r, text) + p.lines[s.header_end:])
    check(source, new)
    return new, s.header_end + 1
