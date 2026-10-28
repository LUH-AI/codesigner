"""`inspect_model_source` checks an uploaded model from its syntax tree, since
importing the upload would run unvetted code in the web process. The background
preparation step later runs the import in the model's own environment.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from dataclasses import dataclass, field

#: `ast.parse` on a large file is real work inside a web request.
MAX_SOURCE_BYTES = 1_000_000

#: Every model environment installs the SDK, and a file listing the SDK gets a
#: confusing index error.
SDK_DISTRIBUTION = "codesigner-model"

#: The upload page, the docs and the error messages share `STARTER_HEADER`.
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

# From PEP 723; captures the `# /// script` body.
PEP723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
_PEP723_OPENING = re.compile(r"(?m)^# /// script$")

_ALWAYS_AVAILABLE = frozenset({"codesigner_model"})


@dataclass(frozen=True)
class ModelSourceInfo:

    #: A literal `name = "..."` in the class body.
    name: str
    class_name: str
    dependencies: list[str] = field(default_factory=list)
    requires_python: str | None = None
    has_header: bool = False
    #: A class that names no tasks is a classifier, as every model was before
    #: tasks.
    tasks: tuple = ("classification",)
    #: Only a forecaster can be backtested, since an upload runs in its own
    #: process.
    forecaster: bool = False
    #: `has_proba` comes from the source, so a run can offer probability metrics
    #: before the file runs.
    has_proba: bool = False


def inspect_model_source(source: bytes) -> tuple[ModelSourceInfo | None, str | None]:
    """The result is (info, None) or (None, error), and the caller wraps the
    untranslated English error.
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
        # Matches the harness's wording for the same failure.
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

    tasks, err = _literal_tasks(model_class)
    if err:
        return None, err

    return ModelSourceInfo(
        name=name,
        class_name=model_class.name,
        tasks=tasks,
        forecaster=_literal_flag(model_class, "forecaster"),
        has_proba=any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == "fit_predict_proba" for node in model_class.body),
        dependencies=deps,
        requires_python=requires_python,
        has_header=has_header,
    ), None


def _find_model_class(tree: ast.Module) -> ast.ClassDef | None:
    """`_find_model_class` matches the trailing base name, since an unimported
    file has nothing to resolve aliases against.
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


def _assigned(cls: ast.ClassDef, attribute: str) -> ast.expr | None:
    """`_assigned` skips annotations without a value."""
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == attribute:
            return node.value
    return None


def _literal_name(cls: ast.ClassDef) -> str | None:
    """A `@property` name needs the file to run, and the experiment needs a name
    while the environment builds.
    """
    node = _assigned(cls, "name")
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) and value.strip() else None


def _literal_tasks(cls: ast.ClassDef) -> tuple[tuple, str | None]:
    from .tasks import TASKS

    node = _assigned(cls, "tasks")
    if node is None:
        return ("classification",), None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return (), (f"'{cls.name}' must set tasks as a literal, "
                    f'for example: tasks = ("classification", "regression").')
    value = (value,) if isinstance(value, str) else tuple(value or ())
    unknown = [t for t in value if t not in TASKS]
    if unknown or not value:
        return (), (f"'{cls.name}' names tasks Codesigner does not know: "
                    f"{', '.join(map(repr, unknown)) or 'none'}. "
                    f"Choose from {', '.join(map(repr, TASKS))}.")
    return value, None


def _literal_flag(cls: ast.ClassDef, attribute: str) -> bool:
    node = _assigned(cls, attribute)
    if node is None:
        return False
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, SyntaxError):
        return False


def _declared_dependencies(text: str) -> tuple[list[str], str | None, bool, str | None]:
    """`_declared_dependencies` returns the tuple (deps, requires_python,
    present, error)."""
    # The PEP's greedy regex would merge two blocks into one with a stray
    # delimiter, and the user would get a TOML error.
    if len(_PEP723_OPENING.findall(text)) > 1:
        return [], None, True, "The file has more than one `# /// script` block."

    blocks = [m for m in PEP723_BLOCK.finditer(text) if m.group("type") == "script"]
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
    name = re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0]
    return name.replace("_", "-").lower()


def _check_undeclared_imports(tree: ast.Module, deps: list[str],
                              has_header: bool) -> str | None:
    """A model environment holds only what the file declares, so an undeclared
    import would fail minutes later. A present header goes unchecked, because
    import and distribution names differ (`sklearn` from `scikit-learn`).
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
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found
