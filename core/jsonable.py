"""`core.jsonable` makes values JSON-safe."""

from __future__ import annotations

import math
from typing import Any


def finite(value: Any) -> Any:
    """`finite` replaces non-finite floats in *value* with None, recursively.
    `json.dumps` writes them as `Infinity` and `NaN`, which strict parsers
    fail to read, and SMAC's `crash_cost` and `walltime_limit` default to infinity.
    `default=` cannot help, since `json.dumps` never calls it for a float.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def plain(value: Any) -> Any:
    """`plain` unwraps a numpy scalar to its Python value."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value
