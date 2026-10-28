"""`SafeJSONField` stores the non-finite floats in SMAC's scenario state as
sentinel objects, because SQLite's JSON column rejects the `Infinity` token.
"""

import math

from django.db import models

_SENTINEL = "__nonfinite__"
_NAMES = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _encode(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return {_SENTINEL: "nan"}
        return {_SENTINEL: "inf" if obj > 0 else "-inf"}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if list(obj.keys()) == [_SENTINEL]:
            return _NAMES[obj[_SENTINEL]]
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


class SafeJSONField(models.JSONField):
    """`SafeJSONField` keeps inf and nan intact."""

    def get_prep_value(self, value):
        if value is None:
            return value
        return super().get_prep_value(_encode(value))

    def from_db_value(self, value, expression, connection):
        value = super().from_db_value(value, expression, connection)
        return _decode(value) if value is not None else value
