"""Values as a JSON file can carry them."""

from __future__ import annotations

import math
from typing import Any


def finite(value: Any) -> Any:
    """*value* with every non-finite number in it replaced by None, recursively.

    `json.dumps` writes `inf` and `nan` as the bare tokens `Infinity` and `NaN`,
    which Python reads back and a strict parser refuses — so an `.ihpo`
    containing one is not the portable file it claims to be. Two sources put
    them there and neither is a measurement:

    - SMAC's `scenario.json`, carried verbatim in `optimizer_state`, whose
      `crash_cost` and `walltime_limit` default to infinity. Every `.ihpo` ever
      written for a SMAC run carries four of them.
    - a run imported from somebody else's output, where SMAC writes the same
      tokens itself.

    None rather than a large number: infinity is not a value here, it is the
    absence of a limit, and nothing reads these fields back. `default=` on
    `json.dumps` cannot do this — it is never called for a float.
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
    """*value* as a Python value: a numpy scalar as the number or string it holds."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value
