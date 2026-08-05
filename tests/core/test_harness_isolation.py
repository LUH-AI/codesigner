"""The harness runs where this application does not exist.

It is executed inside the model's own environment, which contains the model's
dependencies, numpy and the zero-dependency SDK — and nothing else. An import of
`core` or `ui` would work in the test suite (where everything is on the path) and
fail in production, so it is checked structurally rather than by running it.
"""

import ast
import sys
from pathlib import Path

from core.modelhost import HARNESS, protocol

#: What the harness is allowed to reach for. Everything else is either the
#: model's own business or unavailable where it runs.
ALLOWED = {
    "numpy",              # reads the feature arrays
    "codesigner_model",   # the contract, installed into every model environment
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.add(f"<relative:{node.module}>")
            elif node.module:
                found.add(node.module.split(".")[0])
    return found


def test_the_harness_imports_nothing_from_this_application():
    """No `core`, no `ui`, and no relative imports — it has no package to be
    relative to when it runs."""
    offenders = {
        name for name in _imports(HARNESS)
        if name not in ALLOWED and name not in sys.stdlib_module_names
    }
    assert not offenders, f"the harness cannot import {sorted(offenders)} where it runs"


def test_the_harness_agrees_with_the_protocol_version():
    """It carries its own copy of the number, because it cannot import the one
    in protocol.py — so the two must be checked against each other."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    declared = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PROTOCOL_VERSION"
    )
    assert declared == protocol.PROTOCOL_VERSION


def test_the_harness_is_a_script_not_a_module():
    """It is executed, never imported, so it needs a main guard and must not be
    reachable as `core.modelhost.harness` by accident."""
    source = HARNESS.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
