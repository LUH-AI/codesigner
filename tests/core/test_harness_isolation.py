"""The harness runs in the model's environment, where a `core` import would
fail.
"""

import ast
import sys
from pathlib import Path

from core.modelhost import HARNESS, protocol

#: Imports the harness may use besides the standard library.
ALLOWED = {
    "numpy",
    "codesigner_model",   # in every model environment
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
    """Relative imports fail because the harness runs without a package."""
    offenders = {
        name for name in _imports(HARNESS)
        if name not in ALLOWED and name not in sys.stdlib_module_names
    }
    assert not offenders, f"the harness cannot import {sorted(offenders)} where it runs"


def test_the_harness_agrees_with_the_protocol_version():
    """The harness copies `PROTOCOL_VERSION` because it cannot import
    `protocol.py`."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    declared = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PROTOCOL_VERSION"
        and isinstance(node.value, ast.Constant)
    )
    assert declared == protocol.PROTOCOL_VERSION


def test_the_harness_is_a_script_not_a_module():
    source = HARNESS.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
