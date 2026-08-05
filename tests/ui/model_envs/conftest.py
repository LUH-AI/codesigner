"""A stand-in for uv, so these tests do not need it installed.

The shim is a real executable that gets a real argv, exits with a real status and
writes real files — so what is under test is the whole command-construction,
exit-code and output-parsing path, rather than a mock of it. What it does not
test is whether uv itself accepts those arguments; that needs uv, and lives
behind the `uv` marker.
"""

import json
import os
import stat
import textwrap

import pytest

_SHIM = '''\
import json, sys
from pathlib import Path

argv = sys.argv[1:]
mode = argv[0] if argv else ""
script = None
if "--script" in argv:
    script = Path(argv[argv.index("--script") + 1])

if mode == "--version":
    print("uv 0.0.0-fake")
elif mode == "lock":
    {lock_behaviour}
elif mode == "run":
    {run_behaviour}
sys.exit(0)
'''

_WRITES_LOCK = 'Path(str(script) + ".lock").write_text("# fake lock\\n")'
_GREETS = (
    'print(json.dumps({"t": "hello", "protocol": 1, "name": "Fake Model", '
    '"model_class": "FakeModel", "python": "3.12.0", '
    '"config_space": {"name": "s", "hyperparameters": [], "conditions": [], '
    '"forbiddens": [], "format_version": 0.4, "python_module_version": "1.2.0"}}))'
)


def _write_shim(path, *, lock_behaviour, run_behaviour):
    path.write_text("#!/usr/bin/env python3\n" + _SHIM.format(
        lock_behaviour=lock_behaviour, run_behaviour=run_behaviour))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_uv(tmp_path, settings):
    """A uv that resolves and greets successfully."""
    shim = _write_shim(tmp_path / "uv", lock_behaviour=_WRITES_LOCK, run_behaviour=_GREETS)
    settings.UV_BIN = str(shim)
    return shim


@pytest.fixture
def failing_uv(tmp_path, settings):
    """A uv whose resolution fails, the way a bad dependency name would."""
    shim = _write_shim(
        tmp_path / "uv",
        lock_behaviour='sys.stderr.write("x No solution found: no such package `nope`\\n"); sys.exit(1)',
        run_behaviour=_GREETS)
    settings.UV_BIN = str(shim)
    return shim


@pytest.fixture
def no_uv(tmp_path, settings):
    """No uv on this machine."""
    settings.UV_BIN = str(tmp_path / "definitely-not-here")
