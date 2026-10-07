"""A dataset's feature columns as the float matrix every model is given.

`fit` reads a frame once and writes down what each column needs — its kind
(`core.columns`), and for a categorical one the labels it has — as a plain
JSON-able dict. `encode` turns any frame with those columns into floats:

* numeric: the value, NaN where missing;
* categorical: the label's position among the fitted labels, NaN where missing
  or where the label was never seen;
* datetime: calendar features — year, month, day, weekday, hour when the
  times are not all midnight, and the day count since 1970 for a model that
  wants time as one quantity — NaN where missing or unreadable.

Fitted on the whole dataset, not on a training fold. The labels are the only
thing fitted, and which labels a column has says nothing about the target: a
fold-fitted vocabulary would only make one split's validation rows "unseen"
for a label the next split trains on, and give the same row a different code
depending on the seed.

`encode` is written to stand alone — numpy and pandas, nothing from this
package — because an exported model carries its source and the dict, and
predicts from a raw CSV with nothing else installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import CATEGORICAL, DATETIME, NUMERIC, infer_kinds

#: Calendar features always taken from a datetime column, in output order.
DATE_PARTS = ("year", "month", "day", "weekday", "days")
#: Taken too when some value has a time of day.
TIME_PARTS = ("hour",)


def fit(frame: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict:
    """What `encode` needs to know about *frame*'s columns.

    *kinds* overrides the inferred kind of the columns it names — what a
    person who knows the data has said about it.
    """
    inferred = infer_kinds(frame)
    inferred.update({k: v for k, v in (kinds or {}).items() if k in inferred})
    columns = []
    for name, kind in inferred.items():
        series = frame[name]
        column = {"name": name, "kind": kind}
        if kind == CATEGORICAL:
            column["levels"] = sorted({str(v) for v in series.dropna()})
        elif kind == DATETIME:
            stamps = _datetimes(series).dropna()
            timed = bool(len(stamps)) and bool((stamps != stamps.dt.normalize()).any())
            column["parts"] = list(DATE_PARTS + (TIME_PARTS if timed else ()))
        columns.append(column)
    return {"columns": columns}


def feature_names(spec: dict) -> list[str]:
    """The name of each column `encode` returns, in order."""
    names = []
    for column in spec["columns"]:
        if column["kind"] == DATETIME:
            names.extend(f"{column['name']}:{part}" for part in column["parts"])
        else:
            names.append(column["name"])
    return names


def feature_kinds(spec: dict) -> list[str]:
    """The kind of each column `encode` returns: categorical for a label's code,
    numeric for everything else, a calendar part included."""
    kinds = []
    for column in spec["columns"]:
        if column["kind"] == DATETIME:
            kinds.extend([NUMERIC] * len(column["parts"]))
        else:
            kinds.append(column["kind"])
    return kinds


def is_plain(spec: dict) -> bool:
    """Whether every column is a number already, so `encode` changes nothing
    but the dtype."""
    return all(column["kind"] == NUMERIC for column in spec["columns"])


def _datetimes(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return stamps.dt.tz_convert(None)


def encode(spec, frame):
    """*frame*'s features as floats, one row per row, as *spec* describes.

    Columns are found by name, so a frame may order them differently or carry
    extra ones; a column it lacks is all missing.
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
                # A label column read back as numbers — a file whose copy of
                # it has only numeric labels and a gap — holds 1.0 for "1".
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
