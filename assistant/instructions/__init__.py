"""`load` reads `<name>.md` from `assistant/instructions/`. `assistant.prompts`
gives the model `core.md` plus the stage file for planning or for an
experiment."""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).parent


@cache
def load(name: str) -> str:
    return (HERE / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")
