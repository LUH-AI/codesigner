"""Path checks for names that arrive inside an experiment file.

An `.ihpo` is data from wherever the user got it, and parts of it name files:
`optimizer_state` is keyed by the relative path each optimizer file was written
to. Serialization always produces plain relative keys — `f.relative_to(root)` —
so anything else is a file that was edited to reach outside where we put it.

`Path` joining does not help by itself: an absolute right-hand operand wins
outright, so `Path("/tmp/out") / Path("/etc/passwd")` is `/etc/passwd`, not a
path under /tmp/out.
"""

from pathlib import Path, PurePosixPath

#: Most `.ihpo` files hold a dozen optimizer files; this only exists to stop one
#: claiming to hold a million.
MAX_STATE_FILES = 500


def is_safe_relative(name) -> bool:
    """True when *name* is a plain relative path that stays where it is put."""
    if not isinstance(name, str) or not name:
        return False
    if PurePosixPath(name).is_absolute() or Path(name).is_absolute():
        return False
    return ".." not in PurePosixPath(name).parts


def safe_join(root: Path, name: str) -> Path:
    """*name* resolved beneath *root*, or ValueError if it would escape.

    Checks the resolved path as well as the name, so a symlink already sitting
    under *root* cannot be used as the way out.
    """
    if not is_safe_relative(name):
        raise ValueError(f"unsafe path in experiment file: {name!r}")
    dest = (root / name).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe path in experiment file: {name!r}")
    return dest
