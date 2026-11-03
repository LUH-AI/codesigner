"""The model always gets `core.md`, plus one file per stage."""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).parent


@cache
def load(name: str) -> str:
    """`load` reads *name*.md without its closing newline."""
    return (HERE / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")
