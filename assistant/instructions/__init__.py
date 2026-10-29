"""What the model is told, one Markdown file per part: `core.md` for every
request, and one file for each stage of the conversation."""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).parent


@cache
def load(name: str) -> str:
    """`load` returns the text of the instruction file *name*, without its
    closing newline."""
    return (HERE / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")
