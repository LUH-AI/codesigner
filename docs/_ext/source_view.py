"""Source code in the reference documentation.

Each documented function and class gets its own lines folded under its
docstring, and each module a page holding its whole file, with a switch
between that page and the module's documentation. Names in the code link to
where they are defined: to their documentation where they have it, to the
line that defines them where they do not.

Everything is read from the files with `ast`; nothing is imported.
"""

from __future__ import annotations

import ast
import posixpath
import re
from functools import cache
from pathlib import Path

from docutils import nodes

#: Module name → its file, for every module the reference documents.
MODULES: dict[str, Path] = {}

#: Highlighter classes that name something: a name, an imported module path,
#: a class or function being defined, a decorator, `self` or `cls`.
_NAMES = {"n", "nn", "nc", "nf", "nd", "bp"}
_TOKEN = re.compile(r'<span class="([^"]*)">([^<]*)</span>|<[^>]+>|[^<]+')
_DOT = '<span class="o">.</span>'


def _collect(app):
    """Fill `MODULES` from AutoAPI's directories, migrations aside."""
    MODULES.clear()
    for directory in app.config.autoapi_dirs:
        top = (Path(app.confdir) / directory).resolve()
        for path in sorted(top.rglob("*.py")):
            if "migrations" in path.parts:
                continue
            parts = path.relative_to(top.parent).with_suffix("").parts
            MODULES[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path


def doc_page(module: str) -> str:
    return "autoapi/" + module.replace(".", "/") + "/index"


def source_page(module: str) -> str:
    return "_source/" + module.replace(".", "/")


# ── what a module defines and imports ───────────────────────────────────────

class Module:
    """One module's names, as `ast` reads them."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.is_package = path.name == "__init__.py"
        self.text = path.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()
        #: Qualified name → (first line with decorators, `def`/`class` line, last line).
        self.defs: dict[str, tuple[int, int, int]] = {}
        #: `def`/`class` line → qualified name.
        self.at_line: dict[int, str] = {}
        #: Class qualified name → its own members.
        self.members: dict[str, set[str]] = {}
        #: Module-level assignments → their line.
        self.data: dict[str, int] = {}
        #: Imported name → ("module", name) or ("object", module, name).
        self.imports: dict[str, tuple] = {}
        #: (first line, last line, names local to it) per function.
        self.scopes: list[tuple[int, int, set[str]]] = []
        #: (first line, last line, qualified name) per class.
        self.classes: list[tuple[int, int, str]] = []

        tree = ast.parse(self.text)
        self._visit(tree.body, "")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imports[alias.asname] = ("module", alias.name)
                    else:
                        top = alias.name.split(".")[0]
                        self.imports[top] = ("module", top)
            elif isinstance(node, ast.ImportFrom):
                base = self._absolute(node.module, node.level)
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = ("object", base, alias.name)
        for node in tree.body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            for target in targets:
                if isinstance(target, ast.Name):
                    self.data[target.id] = node.lineno

    def _absolute(self, module: str | None, level: int) -> str:
        if not level:
            return module or ""
        package = self.name.split(".")
        if not self.is_package:
            package = package[:-1]
        package = package[:len(package) - (level - 1)]
        return ".".join(package + ([module] if module else []))

    def _visit(self, body, prefix: str):
        for node in body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            qualname = prefix + node.name
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            assert node.end_lineno is not None
            self.defs[qualname] = (start, node.lineno, node.end_lineno)
            self.at_line[node.lineno] = qualname
            if isinstance(node, ast.ClassDef):
                self.classes.append((node.lineno, node.end_lineno, qualname))
                self.members[qualname] = {
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                } | {t.id for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                     for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                     if isinstance(t, ast.Name)}
                self._visit(node.body, qualname + ".")
            else:
                local = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
                local |= {n.id for n in ast.walk(node)
                          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                self.scopes.append((node.lineno, node.end_lineno, local))

    def local_at(self, name: str, line: int) -> bool:
        """Whether *name* is a parameter or variable of the function around *line*."""
        inside = [s for s in self.scopes if s[0] <= line <= s[1]]
        return bool(inside) and name in min(inside, key=lambda s: s[1] - s[0])[2]

    def class_at(self, line: int) -> str | None:
        inside = [c for c in self.classes if c[0] <= line <= c[1]]
        return min(inside, key=lambda c: c[1] - c[0])[2] if inside else None


@cache
def module(name: str) -> Module | None:
    return Module(name, MODULES[name]) if name in MODULES else None


def resolve(name: str, attr: str, depth: int = 0):
    """What *attr* of module *name* is: ("module", m), ("object", m, qualname),
    or None for anything not documented here. Follows re-exports."""
    if f"{name}.{attr}" in MODULES:
        return ("module", f"{name}.{attr}")
    mod = module(name)
    if mod is None or depth > 8:
        return None
    if attr in mod.defs or attr in mod.data:
        return ("object", name, attr)
    imported = mod.imports.get(attr)
    if imported is None:
        return None
    if imported[0] == "module":
        return ("module", imported[1]) if imported[1] in MODULES else None
    return resolve(imported[1], imported[2], depth + 1)


def member(target, attr: str):
    """*attr* looked up on what *target* resolved to: a module's name, or a
    class's own member."""
    if target is None:
        return None
    if target[0] == "module":
        return resolve(target[1], attr)
    mod = module(target[1])
    if mod is not None and attr in mod.members.get(target[2], set()):
        return ("object", target[1], f"{target[2]}.{attr}")
    return None


def url(target, page: str) -> str | None:
    """Where *target* is shown, relative to *page*: its documentation when it
    has an entry, otherwise the line that defines it."""
    if target is None:
        return None
    if target[0] == "module":
        dest, fragment = doc_page(target[1]), ""
    else:
        _, name, qualname = target
        mod = module(name)
        assert mod is not None
        if any(part.startswith("_") for part in qualname.split(".")):
            line = mod.defs[qualname][1] if qualname in mod.defs else mod.data.get(qualname)
            if line is None:
                return None
            dest, fragment = source_page(name), f"#line-{line}"
        else:
            dest, fragment = doc_page(name), f"#{name}.{qualname}"
    if dest == page:
        return fragment or "#"
    return posixpath.relpath(dest, posixpath.dirname(page)) + ".html" + fragment


# ── names in highlighted code made into links ───────────────────────────────

def linked(html: str, name: str, page: str) -> str:
    """*html*, the highlighted source of module *name*, with every name that
    resolves wrapped in a link to it."""
    mod = module(name)
    if mod is None:
        return html
    tokens = [(m.group(0), m.group(1), m.group(2)) for m in _TOKEN.finditer(html)]
    out, line, i = [], 0, 0
    while i < len(tokens):
        raw, cls, text = tokens[i]
        if cls == "linenos":
            line = int(text)
        if cls not in _NAMES:
            out.append(raw)
            i += 1
            continue
        chain = [i]
        while (i + 2 < len(tokens) and tokens[i + 1][0] == _DOT
               and tokens[i + 2][1] == "n"):
            i += 2
            chain.append(i)
        targets = _resolve_chain(mod, [tokens[k] for k in chain], line)
        for k, index in enumerate(chain):
            if k:
                out.append(_DOT)
            href = url(targets[k], page)
            raw_k = tokens[index][0]
            out.append(f'<a class="source-ref" href="{href}">{raw_k}</a>' if href else raw_k)
        i += 1
    return "".join(out)


def _resolve_chain(mod: Module, chain, line: int) -> list:
    """What each name in a dotted *chain* of highlighted tokens refers to."""
    first_cls, first = chain[0][1], chain[0][2].lstrip("@")
    found: list = [None] * len(chain)
    if first_cls == "nn":
        absolute = mod._absolute(first.lstrip("."), len(first) - len(first.lstrip(".")))
        found[0] = ("module", absolute) if absolute in MODULES else None
        return found
    if first_cls in ("nf", "nc"):
        qualname = mod.at_line.get(line)
        found[0] = ("object", mod.name, qualname) if qualname else None
        return found
    if first_cls == "bp":
        owner = mod.class_at(line)
        if owner and len(chain) > 1:
            found[1] = member(("object", mod.name, owner), chain[1][2])
        return found
    if mod.local_at(first, line):
        return found
    if first in mod.defs or first in mod.data:
        target = ("object", mod.name, first)
    else:
        imported = mod.imports.get(first)
        if imported is None:
            return found
        target = (("module", imported[1]) if imported[0] == "module" and imported[1] in MODULES
                  else resolve(imported[1], imported[2]) if imported[0] == "object" else None)
    found[0] = target
    for k in range(1, len(chain)):
        target = member(target, chain[k][2])
        found[k] = target
    return found


def highlighted(app, code: str, name: str, page: str, start: int = 1) -> str:
    html = app.builder.highlighter.highlight_block(
        code, "python", linenos="inline", linenostart=start, lineanchors="line")
    return f'<div class="highlight-python notranslate">{linked(html, name, page)}</div>'


# ── where the source goes ───────────────────────────────────────────────────

def _switch(docs: str | None, source: str | None) -> str:
    """The Documentation | Source switch; the side being shown is not a link."""
    def side(label, href):
        return f'<a href="{href}">{label}</a>' if href else f"<strong>{label}</strong>"
    return (f'<p class="source-switch">{side("Documentation", docs)}'
            f'<span>|</span>{side("Source", source)}</p>')


def _on_doctree(app, doctree):
    """On a module's documentation page: the switch under its title, and each
    function's and class's own source folded under its docstring."""
    page = app.env.docname
    if not (page.startswith("autoapi/") and page.endswith("/index")):
        return
    name = page[len("autoapi/"):-len("/index")].replace("/", ".")
    mod = module(name)
    if mod is None:
        return

    section = next(iter(doctree.findall(nodes.section)), None)
    if section is not None:
        href = posixpath.relpath(source_page(name), posixpath.dirname(page)) + ".html"
        section.insert(1, nodes.raw("", _switch(None, href), format="html"))

    descs = [d for d in doctree.findall(nodes.Element)
             if d.tagname == "desc" and d.get("domain") == "py"
             and d.get("objtype") in ("function", "method", "class", "exception")]
    for desc in descs:
        signature = desc.children[0]
        own = module(signature.get("module") or "")
        found = own.defs.get(signature.get("fullname") or "") if own else None
        if own is None or found is None:
            continue
        start, _def_line, end = found
        code = "\n".join(own.lines[start - 1:end])
        content = desc.children[-1]
        at = next((i for i, child in enumerate(content.children) if child.tagname == "desc"),
                  len(content.children))
        folded = nodes.raw("", '<details class="source"><summary>Source code</summary>'
                           + highlighted(app, code, own.name, page, start)
                           + "</details>", format="html")
        content.insert(at, folded)


def _source_pages(app):
    """One page per module with its whole source, and the switch back."""
    for name in MODULES:
        page = source_page(name)
        docs = posixpath.relpath(doc_page(name), posixpath.dirname(page)) + ".html"
        mod = module(name)
        assert mod is not None
        body = (f"<h1>{name}</h1>" + _switch(docs, None)
                + highlighted(app, mod.text, name, page))
        yield page, {"title": name, "body": body}, "page.html"


def setup(app):
    app.connect("builder-inited", _collect)
    app.connect("doctree-read", _on_doctree)
    app.connect("html-collect-pages", _source_pages)
    return {"parallel_read_safe": True}
