from __future__ import annotations

from functools import cache
from pathlib import Path


@cache
def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _source(app, full_name: str) -> Path | None:
    parts = full_name.split(".")
    for directory in app.config.autoapi_dirs:
        root = (Path(app.confdir) / directory).resolve()
        if parts[0] != root.name:
            continue
        for end in range(len(parts) - 1, 0, -1):
            base = root.joinpath(*parts[1:end])
            for candidate in (base.with_suffix(".py") if end > 1 else None,
                              base / "__init__.py"):
                if candidate is not None and candidate.is_file():
                    return candidate
    return None


def _comment_above(lines: list[str], line_no: int) -> str:
    found = []
    index = line_no - 2
    if index >= len(lines):
        return ""
    while index >= 0 and lines[index].lstrip().startswith("#:"):
        found.append(lines[index].lstrip()[2:].strip())
        index -= 1
    return "\n".join(reversed(found))


def _attach(app, what, name, obj, skip, options):
    if (what not in ("data", "attribute") or obj.docstring
            or getattr(obj, "imported", False) or getattr(obj, "inherited", False)):
        return None
    line_no = obj.obj.get("from_line_no")
    path = _source(app, name)
    if line_no and path is not None:
        obj.docstring = _comment_above(_lines(path), line_no)
    return None


def setup(app):
    app.connect("autoapi-skip-member", _attach)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
