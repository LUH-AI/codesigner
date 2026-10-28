"""`optimizer_state` keys in an `.ihpo` are plain relative paths, so anything
else marks a tampered file. A `Path` join lets an absolute right operand win.
"""

from pathlib import Path, PurePosixPath

#: `MAX_STATE_FILES` stops a file claiming a million entries.
MAX_STATE_FILES = 500


def is_safe_relative(name) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if PurePosixPath(name).is_absolute() or Path(name).is_absolute():
        return False
    return ".." not in PurePosixPath(name).parts


def safe_join(root: Path, name: str) -> Path:
    """`safe_join` also checks the resolved path, so a symlink under *root*
    cannot lead out.
    """
    if not is_safe_relative(name):
        raise ValueError(f"unsafe path in experiment file: {name!r}")
    dest = (root / name).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe path in experiment file: {name!r}")
    return dest
