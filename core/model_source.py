"""Reading a model file without running it.

Finding out whether an uploaded `.py` is a usable model used to mean importing
it, which executes everything in it. That happened while the user waited on the
form, in the web process — so an upload could hold a worker for as long as its
imports took, and the app's own environment ran code it had not vetted.

Almost everything the form needs is visible in the source text: does it parse,
is there a model class, what is it called, and what does it say it needs. This
module answers those from the syntax tree. What genuinely requires execution —
that the file *imports* cleanly and that the class instantiates — is left to the
background preparation step, which runs it in its own environment where it
belongs.

Takes bytes rather than a path: the caller has the upload in hand and should not
have to write a temp file for a syntax check.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from dataclasses import dataclass, field

#: Refuse anything implausible as a model before parsing it. `ast.parse` on a
#: large file is real work, and this runs in a web request.
MAX_SOURCE_BYTES = 1_000_000

#: The distribution that provides `BaseModel`. It is installed into every model
#: environment automatically, so a file that also lists it gets a confusing
#: "no such package" from the index.
SDK_DISTRIBUTION = "codesigner-model"

#: What to paste at the top of a model file. Exported so the upload page, the
#: docs and the error messages cannot drift apart.
STARTER_HEADER = """\
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "numpy",
#     "ConfigSpace",
# ]
# ///
"""

# From PEP 723. Matches the `# /// script` block and captures its body.
_PEP723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
_PEP723_OPENING = re.compile(r"(?m)^# /// script$")

_ALWAYS_AVAILABLE = frozenset({"codesigner_model"})


@dataclass(frozen=True)
class ModelSourceInfo:
    """What a model file says about itself, read from its source."""

    #: The model's display name — a literal `name = "..."` in the class body.
    name: str
    #: The class implementing the model.
    class_name: str
    #: Dependencies declared in the PEP 723 header, verbatim.
    dependencies: list[str] = field(default_factory=list)
    #: The header's `requires-python`, if it gave one.
    requires_python: str | None = None
    #: Whether the file carried a PEP 723 header at all.
    has_header: bool = False


def inspect_model_source(source: bytes) -> tuple[ModelSourceInfo | None, str | None]:
    """Read *source* as a model file. Returns (info, None) or (None, error).

    Errors are plain English and deliberately untranslated, matching the rest of
    the loading path — the caller wraps them in a translated sentence.
    """
    if len(source) > MAX_SOURCE_BYTES:
        return None, (f"That file is too large to be a model "
                      f"(limit {MAX_SOURCE_BYTES // 1000} kB).")

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        return None, "The file is not valid UTF-8 text."

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return None, f"The file is not valid Python: {exc.msg} (line {exc.lineno})."

    model_class = _find_model_class(tree)
    if model_class is None:
        # Kept word-for-word from the loader this replaces, so the message a
        # user sees does not depend on which check caught it.
        return None, "No BaseModel subclass found in the file."

    name = _literal_name(model_class)
    if name is None:
        return None, (f"'{model_class.name}' must set a literal name, "
                      f'for example: name = "My Model".')

    deps, requires_python, has_header, err = _declared_dependencies(text)
    if err:
        return None, err

    if err := _check_undeclared_imports(tree, deps, has_header):
        return None, err

    return ModelSourceInfo(
        name=name,
        class_name=model_class.name,
        dependencies=deps,
        requires_python=requires_python,
        has_header=has_header,
    ), None


def _find_model_class(tree: ast.Module) -> ast.ClassDef | None:
    """The first top-level class inheriting something called BaseModel.

    Matched on the trailing name, so `BaseModel`, `codesigner_model.BaseModel`
    and an aliased import all count — the file is not imported, so there is
    nothing to resolve the reference against.
    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
                _base_name(base) == "BaseModel" for base in node.bases):
            return node
    return None


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_name(cls: ast.ClassDef) -> str | None:
    """The class's `name = "..."`, if it is a plain string literal.

    A `@property def name` is legal in the contract but cannot be read without
    running the file, and the experiment needs something to be called while its
    environment is still being prepared. Preparation reports the instantiated
    model's real name later and corrects the record if they differ.
    """
    for node in cls.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if not isinstance(target, ast.Name) or target.id != "name":
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(value, str) and value.strip():
            return value
    return None


def _declared_dependencies(text: str) -> tuple[list[str], str | None, bool, str | None]:
    """Parse the PEP 723 header. Returns (deps, requires_python, present, error)."""
    # Counted separately from the match below: the PEP's regex is greedy, so two
    # blocks match as one whose body contains a stray delimiter, and the user
    # would get a TOML syntax error instead of being told what they did.
    if len(_PEP723_OPENING.findall(text)) > 1:
        return [], None, True, "The file has more than one `# /// script` block."

    blocks = [m for m in _PEP723_BLOCK.finditer(text) if m.group("type") == "script"]
    if not blocks:
        return [], None, False, None

    body = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in blocks[0].group("content").splitlines(keepends=True)
    )
    try:
        meta = tomllib.loads(body)
    except tomllib.TOMLDecodeError as exc:
        return [], None, True, f"The `# /// script` header is not valid TOML: {exc}"

    deps = meta.get("dependencies", [])
    if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
        return [], None, True, "The header's `dependencies` must be a list of strings."

    if any(_requirement_name(d) == SDK_DISTRIBUTION for d in deps):
        return [], None, True, (f"Remove `{SDK_DISTRIBUTION}` from dependencies — "
                                f"it is provided automatically.")

    requires_python = meta.get("requires-python")
    if requires_python is not None and not isinstance(requires_python, str):
        return [], None, True, "The header's `requires-python` must be a string."

    return deps, requires_python, True, None


def _requirement_name(requirement: str) -> str:
    """The distribution name from a requirement string, lowercased."""
    name = re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0]
    return name.replace("_", "-").lower()


def _check_undeclared_imports(tree: ast.Module, deps: list[str],
                              has_header: bool) -> str | None:
    """Catch a file with no header that imports something it will not have.

    Only when there is no header at all. A model environment contains exactly
    what the file declares, so such an import is certain to fail — several
    minutes later, once the environment has been built. Saying so now turns a
    slow, confusing failure into an immediate one.

    A header that *is* present is never second-guessed: import names and
    distribution names disagree often enough (`sklearn` from `scikit-learn`,
    `cv2` from `opencv-python`) that checking them would reject working files.
    """
    if has_header or deps:
        return None

    for module in sorted(_toplevel_imports(tree)):
        if module in sys.stdlib_module_names or module in _ALWAYS_AVAILABLE:
            continue
        return (f"This model imports `{module}`, but declares no dependencies. "
                f"Your model runs in its own environment, so add a header "
                f"saying what it needs:\n\n{STARTER_HEADER}")
    return None


def _toplevel_imports(tree: ast.Module) -> set[str]:
    """Every root package the module imports, at any nesting level."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found
