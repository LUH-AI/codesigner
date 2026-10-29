"""`fit` records each column's kind and categorical labels as a JSON-able dict,
and `encode` turns any frame with those columns into floats, NaN for a missing
or unseen value. `fit` sees the whole dataset, since a fold-fitted vocabulary
would code one row differently per seed. `encode` only uses numpy and pandas,
because an exported model carries its source and predicts from a raw CSV.
"""

from __future__ import annotations

import pandas as pd

from .columns import CATEGORICAL, DATETIME, NUMERIC, infer_kinds

#: Always taken, in output order.
DATE_PARTS = ("year", "month", "day", "weekday", "days")
#: Added when any value has a time.
TIME_PARTS = ("hour",)


def fit(frame: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict:
    """`fit` takes *kinds* to override inferred column kinds.
    """
    inferred = infer_kinds(frame)
    inferred.update({k: v for k, v in (kinds or {}).items() if k in inferred})
    columns = []
    for name, kind in inferred.items():
        series = frame[name]
        column: dict[str, str | list[str]] = {"name": name, "kind": kind}
        if kind == CATEGORICAL:
            column["levels"] = sorted({str(v) for v in series.dropna()})
        elif kind == DATETIME:
            stamps = _datetimes(series).dropna()
            timed = bool(len(stamps)) and bool((stamps != stamps.dt.normalize()).any())
            column["parts"] = list(DATE_PARTS + (TIME_PARTS if timed else ()))
        columns.append(column)
    return {"columns": columns}


def feature_names(spec: dict) -> list[str]:
    names = []
    for column in spec["columns"]:
        if column["kind"] == DATETIME:
            names.extend(f"{column['name']}:{part}" for part in column["parts"])
        else:
            names.append(column["name"])
    return names


def feature_kinds(spec: dict) -> list[str]:
    """Every datetime part counts as numeric."""
    kinds = []
    for column in spec["columns"]:
        if column["kind"] == DATETIME:
            kinds.extend([NUMERIC] * len(column["parts"]))
        else:
            kinds.append(column["kind"])
    return kinds


def is_plain(spec: dict) -> bool:
    """`is_plain` is True when every column is numeric already."""
    return all(column["kind"] == NUMERIC for column in spec["columns"])


def _datetimes(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return stamps.dt.tz_convert(None)


def encode(spec, frame):
    """`encode` finds columns by name and fills a missing column with NaN
    values.
    """
    import warnings

    import numpy as np
    import pandas as pd

    def parse_dates(values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stamps = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
        return stamps.dt.tz_convert(None)

    n = len(frame)
    out = []
    for column in spec["columns"]:
        name, kind = column["name"], column["kind"]
        values = frame[name] if name in frame.columns else pd.Series([np.nan] * n, index=frame.index)
        if kind == "categorical":
            position = {level: float(i) for i, level in enumerate(column["levels"])}

            def code(v, position=position):
                # A numeric-only label column read back as floats contains 1.0 for
                # "1".
                text = str(v)
                if text not in position and isinstance(v, float) and v.is_integer():
                    text = str(int(v))
                return position.get(text, np.nan)

            out.append(np.array([np.nan if pd.isna(v) else code(v) for v in values], dtype=float))
        elif kind == "datetime":
            stamps = parse_dates(values)
            parts = {
                "year": stamps.dt.year, "month": stamps.dt.month, "day": stamps.dt.day,
                "weekday": stamps.dt.weekday, "hour": stamps.dt.hour,
                "days": (stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1),
            }
            out.extend(np.asarray(parts[part], dtype=float) for part in column["parts"])
        else:
            out.append(np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float))
    return np.column_stack(out) if out else np.empty((n, 0))
