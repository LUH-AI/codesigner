"""`source_view` adds source code to the reference documentation.

Each documented function and class gets its own lines folded under its
docstring, and each module a page containing its whole file, with a switch
between that page and the module's documentation. Names in the code link to
where they are defined: codesigner's own to their documentation (or, when
private, the line that defines them), and other projects' into their
documentation through intersphinx.

What a name refers to is asked of Jedi, which infers types statically; where
it has no answer, the module's own imports and definitions, read with `ast`,
decide. Nothing is imported.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import posixpath
import re
from functools import cache
from pathlib import Path

from docutils import nodes

#: Module name → its file, for every module the reference documents.
MODULES: dict[str, Path] = {}
#: The reverse, by resolved path.
BY_PATH: dict[Path, str] = {}

#: Highlighter classes that name something: a name, an imported module path,
#: a class or function being defined, a decorator, `self` or `cls`.
_NAMES = {"n", "nn", "nc", "nf", "nd", "bp"}
_TOKEN = re.compile(r'<span class="([^"]*)">([^<]*)</span>|<[^>]+>|[^<]+')
_DOT = '<span class="o">.</span>'

#: A name Jedi found to be a parameter or local variable: never linked, and
#: not handed to the fallback either.
LOCAL = ("local",)


def _collect(app):
    """Fill `MODULES` from AutoAPI's directories, migrations aside, and open
    the cache of what Jedi said last time."""
    MODULES.clear()
    BY_PATH.clear()
    for directory in app.config.autoapi_dirs:
        top = (Path(app.confdir) / directory).resolve()
        for path in sorted(top.rglob("*.py")):
            if "migrations" in path.parts:
                continue
            parts = path.relative_to(top.parent).with_suffix("").parts
            name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            MODULES[name] = path
            BY_PATH[path] = name
    _Resolver.open(app)


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
        #: A class's fields and class attributes, by qualified name → their line.
        self.fields: dict[str, int] = {}
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
                base = self.absolute(node.module, node.level)
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = ("object", base, alias.name)
        for node in tree.body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            for target in targets:
                if isinstance(target, ast.Name):
                    self.data[target.id] = node.lineno

    def absolute(self, module: str | None, level: int) -> str:
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
                for n in node.body:
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                            if isinstance(t, ast.Name):
                                self.fields[f"{qualname}.{t.id}"] = n.lineno
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

    def documents(self, qualname: str) -> bool:
        """Whether the reference has an entry for *qualname* in this module."""
        if any(part.startswith("_") for part in qualname.split(".")):
            return False
        if qualname in self.defs or qualname in self.data:
            return True
        owner, _, attr = qualname.rpartition(".")
        return attr in self.members.get(owner, set())

    def line_of(self, qualname: str) -> int | None:
        if qualname in self.defs:
            return self.defs[qualname][1]
        return self.data.get(qualname, self.fields.get(qualname))


@cache
def _digest(name: str) -> str:
    return hashlib.sha256(MODULES[name].read_bytes()).hexdigest()


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
    if target is None or target[0] not in ("module", "object"):
        return None
    if target[0] == "module":
        return resolve(target[1], attr)
    mod = module(target[1])
    if mod is not None and attr in mod.members.get(target[2], set()):
        return ("object", target[1], f"{target[2]}.{attr}")
    return None


def _static_chain(mod: Module, chain, line: int) -> list:
    """What each name in a dotted *chain* of highlighted tokens refers to, by
    the module's own imports and definitions alone."""
    first_cls, first = chain[0][1], chain[0][2].lstrip("@")
    found: list = [None] * len(chain)
    if first_cls == "nn":
        absolute = mod.absolute(first.lstrip("."), len(first) - len(first.lstrip(".")))
        found[0] = (("module", absolute) if absolute in MODULES
                    else _Resolver.external(absolute))
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


# ── what Jedi says a name is ────────────────────────────────────────────────

class _Resolver:
    """Jedi's answers, per module, kept between builds in `docs/_build` and
    reused while the module is unchanged."""

    project = None
    scripts: dict = {}
    answers: dict = {}
    #: This build's checked cache entry per module.
    current: dict = {}
    cache_path: Path | None = None
    fingerprint = ""
    inventory: dict[str, str] = {}

    @classmethod
    def open(cls, app):
        import jedi

        root = Path(app.confdir).parent
        environment = root / ".venv"
        cls.project = jedi.Project(
            root, added_sys_path=[str(root / "model_sdk")],
            **({"environment_path": str(environment)} if environment.is_dir() else {}))
        cls.scripts = {}
        cls.current = {}
        cls.cache_path = Path(app.confdir) / "_build" / "source_view.json"
        # Answers name other projects' pages, so a change to where those are
        # looked up starts the cache over.
        cls.fingerprint = json.dumps(app.config.intersphinx_mapping, sort_keys=True, default=str)
        try:
            stored = json.loads(cls.cache_path.read_text())
        except (OSError, ValueError):
            stored = {}
        cls.answers = (stored.get("modules", {})
                       if stored.get("fingerprint") == cls.fingerprint else {})
        cls.inventory = {}

    @classmethod
    def save(cls, app, exception):
        if exception is None and cls.cache_path is not None:
            cls.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cls.cache_path.write_text(json.dumps(
                {"fingerprint": cls.fingerprint, "modules": cls.answers}))

    @classmethod
    def load_inventory(cls, app):
        """Every object other projects document, by name, from intersphinx."""
        inventory = getattr(app.env, "intersphinx_inventory", {}) or {}
        for objtype, entries in inventory.items():
            if objtype.startswith("py:"):
                for name, item in entries.items():
                    cls.inventory.setdefault(name, item.uri)

    @classmethod
    def external(cls, full_name: str | None):
        """("external", url) for *full_name* if another project documents it.
        A definition's own path often differs from the one documented, so the
        names it is re-exported under are tried too: `pandas.core.frame.DataFrame`
        as `pandas.DataFrame`."""
        if not full_name:
            return None
        if full_name.startswith("builtins."):
            full_name = full_name[len("builtins."):]
        parts = full_name.split(".")
        candidates = [full_name] + [".".join([parts[0]] + parts[k:]) for k in range(1, len(parts) - 1)]
        for candidate in candidates:
            if candidate in cls.inventory:
                return ("external", cls.inventory[candidate])
        return None

    @classmethod
    def entry(cls, mod: Module) -> dict:
        if mod.name not in cls.current:
            entry = cls.answers.get(mod.name)
            if entry is None or entry.get("digest") != _digest(mod.name) \
                    or not cls._still_true(entry):
                entry = cls.answers[mod.name] = {"digest": _digest(mod.name), "at": {}}
            cls.current[mod.name] = entry
        return cls.current[mod.name]

    @classmethod
    def _still_true(cls, entry) -> bool:
        """Whether every codesigner object a cached answer named is still
        where it was."""
        for target in entry["at"].values():
            if target and target[0] == "object":
                other = module(target[1])
                if other is None or other.line_of(target[2]) is None:
                    return False
            elif target and target[0] == "module" and target[1] not in MODULES:
                return False
            elif target and target[0] == "line" and (
                    target[1] not in MODULES or _digest(target[1]) != target[3]):
                return False
        return True

    @classmethod
    def at(cls, mod: Module, line: int, column: int):
        """What the name at *line*, *column* of *mod* refers to: a target,
        `LOCAL`, or None when Jedi cannot say."""
        entry = cls.entry(mod)
        key = f"{line}:{column}"
        if key not in entry["at"]:
            entry["at"][key] = cls._ask(mod, line, column)
        answer = entry["at"][key]
        return tuple(answer) if answer else None

    @classmethod
    def _ask(cls, mod: Module, line: int, column: int):
        import jedi

        if mod.name not in cls.scripts:
            cls.scripts[mod.name] = jedi.Script(mod.text, path=MODULES[mod.name],
                                                project=cls.project)
        try:
            found = cls.scripts[mod.name].goto(line, column, follow_imports=True)
        except Exception:  # noqa: BLE001 — an unanswerable name is just not linked
            return None
        if not found:
            return None
        definition = found[0]
        parent = definition.parent()
        if definition.type == "param" or (parent is not None and parent.type == "function"
                                          and definition.type != "function"):
            return list(LOCAL)
        path = definition.module_path
        name = BY_PATH.get(Path(path).resolve()) if path else None
        if name is not None:
            if definition.type == "module":
                return ["module", name]
            full = definition.full_name or ""
            qualname = full[len(name) + 1:] if full.startswith(name + ".") else None
            other = module(name)
            if qualname and other is not None and other.line_of(qualname) is not None:
                return ["object", name, qualname]
            return (["line", name, definition.line, _digest(name)]
                    if definition.line else None)
        target = cls.external(definition.full_name)
        return list(target) if target else None


# ── names in highlighted code made into links ───────────────────────────────

def url(target, page: str) -> str | None:
    """Where *target* is shown, relative to *page*: codesigner's documentation
    entry when it has one, otherwise its line; another project's own page."""
    if target is None or target == LOCAL:
        return None
    if target[0] == "external":
        return target[1]
    if target[0] == "module":
        dest, fragment = doc_page(target[1]), ""
    elif target[0] == "line":
        dest, fragment = source_page(target[1]), f"#line-{target[2]}"
    else:
        _, name, qualname = target
        mod = module(name)
        if mod is None:
            return None
        if mod.documents(qualname):
            dest, fragment = doc_page(name), f"#{name}.{qualname}"
        else:
            line = mod.line_of(qualname)
            if line is None:
                return None
            dest, fragment = source_page(name), f"#line-{line}"
    if dest == page:
        return fragment or "#"
    return posixpath.relpath(dest, posixpath.dirname(page)) + ".html" + fragment


def linked(html_code: str, name: str, page: str) -> str:
    """*html_code*, the highlighted source of module *name*, with every name
    that resolves wrapped in a link to it."""
    mod = module(name)
    if mod is None:
        return html_code
    tokens = []
    line = column = 0
    for match in _TOKEN.finditer(html_code):
        raw, cls, text = match.group(0), match.group(1), match.group(2)
        if cls == "linenos":
            line, column = int(text), 0
            tokens.append((raw, cls, text, line, column))
            continue
        tokens.append((raw, cls, text, line, column))
        if not raw.startswith("<") or cls is not None:
            column += len(html.unescape(text if cls is not None else raw))

    out, i = [], 0
    while i < len(tokens):
        raw, cls, text, line, column = tokens[i]
        if cls not in _NAMES:
            out.append(raw)
            i += 1
            continue
        chain = [i]
        while (i + 2 < len(tokens) and tokens[i + 1][0] == _DOT
               and tokens[i + 2][1] == "n"):
            i += 2
            chain.append(i)
        static = _static_chain(mod, [tokens[k][:3] for k in chain], line)
        for k, index in enumerate(chain):
            if k:
                out.append(_DOT)
            t_raw, t_cls, t_text, t_line, t_column = tokens[index]
            target = None
            if t_cls not in ("nn", "bp"):
                offset = len(t_text) - len(t_text.lstrip("@"))
                target = _Resolver.at(mod, t_line, t_column + offset)
            if target is None:
                target = static[k]
            href = url(target, page)
            out.append(f'<a class="source-ref" href="{html.escape(href)}">{t_raw}</a>'
                       if href else t_raw)
        i += 1
    return "".join(out)


def highlighted(app, code: str, name: str, page: str, start: int = 1) -> str:
    html_code = app.builder.highlighter.highlight_block(
        code, "python", linenos="inline", linenostart=start, lineanchors="line")
    return f'<div class="highlight-python notranslate">{linked(html_code, name, page)}</div>'


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
    if not _Resolver.inventory:
        _Resolver.load_inventory(app)
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
    if not _Resolver.inventory:
        _Resolver.load_inventory(app)
    for name in MODULES:
        page = source_page(name)
        docs = posixpath.relpath(doc_page(name), posixpath.dirname(page)) + ".html"
        mod = module(name)
        assert mod is not None
        body = (f"<h1>{name}</h1>" + _switch(docs, None)
                + highlighted(app, mod.text, name, page))
        yield page, {"title": name, "body": body}, "page.html"


def setup(app):
    app.setup_extension("sphinx.ext.intersphinx")
    app.connect("builder-inited", _collect)
    app.connect("doctree-read", _on_doctree)
    app.connect("html-collect-pages", _source_pages)
    app.connect("build-finished", _Resolver.save)
    return {"parallel_read_safe": False}
