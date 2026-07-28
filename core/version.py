"""The single application version.

The canonical value lives in `pyproject.toml` (`[project].version`) — the one
place a release is bumped. It is read here from the installed package metadata
so nothing in the codebase carries a second copy: `core` stamps it into every
`.ihpo` file as provenance and the `ui` layer displays it, both reading the
same number. `core` depends only on the stdlib for this (no Django), keeping
the domain layer standalone.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    VERSION = version("codesigner")
except PackageNotFoundError:  # running from an uninstalled source tree
    VERSION = "0.0.0+unknown"
