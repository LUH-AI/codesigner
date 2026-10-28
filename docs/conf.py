"""Sphinx configuration for codesigner's reference documentation.

AutoAPI reads the source rather than importing it, so the build needs none of
the application's dependencies, its Django settings or the SMAC branch.
"""

import ast
from functools import cache
from pathlib import Path

from docutils import nodes

ROOT = Path(__file__).resolve().parent.parent

project = "Codesigner"

extensions = ["autoapi.extension", "myst_parser"]

autoapi_type = "python"
autoapi_dirs = ["../access", "../config", "../core", "../experiments", "../remote", "../ui",
                "../model_sdk/codesigner_model"]
autoapi_ignore = ["*migrations*"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_member_order = "bysource"
autoapi_add_toctree_entry = False
autoapi_keep_files = False

# Docstrings write names in single backticks; read them as references to
# Python objects, linked where they resolve.
default_role = "py:obj"

html_theme = "furo"
html_title = "Codesigner"
exclude_patterns = ["_build"]
html_static_path = ["_static"]
html_css_files = ["source.css"]


# ── each object's own source, folded under it ───────────────────────────────

@cache
def _definitions(path: Path) -> dict:
    """Each function and class in *path* by qualified name, as its lines
    (decorators included) and the line it starts on."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = {}

    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                found[prefix + node.name] = (lines[start - 1:node.end_lineno], start)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")

    visit(ast.parse(text).body, "")
    return found


def _module_file(module: str) -> Path | None:
    parts = module.split(".")
    base = ROOT / "model_sdk" if parts[0] == "codesigner_model" else ROOT
    for path in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
        if path.is_file():
            return path
    return None


def _fold_in_source(app, doctree):
    """Add a collapsed copy of each documented function's and class's own
    source under its docstring, above anything documented inside it."""
    descs = [d for d in doctree.findall(nodes.Element)
             if d.tagname == "desc" and d.get("domain") == "py"
             and d.get("objtype") in ("function", "method", "class", "exception")]
    for desc in descs:
        signature = desc.children[0]
        path = _module_file(signature.get("module") or "")
        found = _definitions(path).get(signature.get("fullname") or "") if path else None
        if not found:
            continue
        lines, start = found
        code = "\n".join(lines)
        block = nodes.literal_block(code, code, language="python", linenos=True,
                                    highlight_args={"linenostart": start})
        content = desc.children[-1]
        at = next((i for i, child in enumerate(content.children) if child.tagname == "desc"),
                  len(content.children))
        content.children[at:at] = [
            nodes.raw("", '<details class="source"><summary>Source code</summary>', format="html"),
            block,
            nodes.raw("", "</details>", format="html"),
        ]
        for child in content.children[at:at + 3]:
            child.parent = content


def setup(app):
    app.connect("doctree-read", _fold_in_source)
