"""Questions about a dataset, answered from the data: worked out here, never by
code the model writes.

Each analysis is a fixed operation on the dataset as Codesigner reads it
(`core.datasets.read_dataset`, the predicted column last), with its arguments
checked: a column must be one of the dataset's, an operator one of a short
list. What goes back to the model is aggregates — counts, shares, statistics.
Values of the data themselves (a category's name, a row) only as far as
`ASSISTANT_SAMPLE_ROWS` allows: with it at 0, nothing of the data leaves this
machine but names, kinds and numbers about them. Charts are drawn for the
person, in their browser, and are not what the model is told.

A large dataset is sampled, with a fixed seed, for what costs time on every
row (associations, correlations); counts and shares are always over all rows.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from core import columns as kinds
from core import datasets, tasks

#: Above this many rows, associations and correlations are worked out on a
#: sample of this many.
SAMPLE_ROWS = 50_000
#: Most groups a query answers with.
MAX_GROUPS = 50
#: Numeric columns grouped by are cut into this many ranges of equal counts.
BINS = 5
#: Fewer distinct values than this, and a numeric column is grouped by value.
FEW_VALUES = 20
#: How many frames are kept in memory between questions.
CACHED = 4

OPERATORS = ("=", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_missing", "not_missing")
AGGREGATES = ("count", "mean", "median", "min", "max", "sum", "std", "nunique", "share")


class DataError(ValueError):
    """A question about the data that cannot be asked as put: a column the
    dataset does not have, an operator or aggregate not on the list."""


# ── reading ──────────────────────────────────────────────────────────────────

_frames: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_lock = threading.Lock()


def load(path, target: str = "") -> pd.DataFrame:
    """The dataset at *path* with *target* (the last column when "") last,
    kept for the next question while the file is unchanged."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, target or "")
    with _lock:
        if key in _frames:
            _frames.move_to_end(key)
            return _frames[key]
    frame = datasets.read_dataset(path, target or None)
    with _lock:
        _frames[key] = frame
        while len(_frames) > CACHED:
            _frames.popitem(last=False)
    return frame


def sampled(frame: pd.DataFrame, rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """*frame*, or a seeded sample of *rows* of it when it is larger."""
    return frame if len(frame) <= rows else frame.sample(rows, random_state=0)


def target_name(frame: pd.DataFrame) -> str:
    return str(frame.columns[-1])


def column_of(frame: pd.DataFrame, name) -> pd.Series:
    """The column called *name*, or `DataError` naming the ones there are."""
    for column in frame.columns:
        if str(column) == str(name):
            return frame[column]
    raise DataError(f"there is no column {name!r}; there are {[str(c) for c in frame.columns]}")


def kind(series: pd.Series) -> str:
    return kinds.kind_of(series)


def is_classes(series: pd.Series) -> bool:
    """Whether *series*, as a target, reads as classes: not numbers, or few
    whole numbers."""
    present = series.dropna()
    if not tasks.is_numeric(present.to_numpy()):
        return True
    numbers = pd.to_numeric(present, errors="coerce").dropna()
    return bool(len(numbers)) and bool(np.all(np.mod(numbers, 1) == 0)) \
        and numbers.nunique() <= FEW_VALUES


def _numbers(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _plain(value):
    """A value as JSON has it."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _label(value, samples: int, index: int) -> str:
    """A category as the model may see it: itself, or a stand-in when no values
    of the data may leave this machine."""
    return str(value) if samples else f"value {index + 1}"


# ── one column ───────────────────────────────────────────────────────────────

def describe_column(frame: pd.DataFrame, name, samples: int) -> tuple[dict, dict | None]:
    """What one column holds, and a chart of it: a histogram for numbers, the
    commonest values for categories, the range for dates."""
    series = column_of(frame, name)
    present = series.dropna()
    out = {"column": str(name), "kind": kind(series), "rows": int(len(series)),
           "missing": int(series.isna().sum()), "distinct": int(present.nunique())}
    chart = None
    if out["kind"] == kinds.NUMERIC:
        numbers = _numbers(present).dropna()
        if len(numbers):
            q = numbers.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            iqr = q[0.75] - q[0.25]
            outliers = ((numbers < q[0.25] - 1.5 * iqr) | (numbers > q[0.75] + 1.5 * iqr)).sum()
            out.update(mean=_plain(numbers.mean()), std=_plain(numbers.std()),
                       min=_plain(numbers.min()), max=_plain(numbers.max()),
                       quantiles={f"{int(k * 100)}%": _plain(v) for k, v in q.items()},
                       skew=_plain(numbers.skew()), outliers=int(outliers))
            counts, edges = np.histogram(numbers, bins=min(20, max(1, int(numbers.nunique()))))
            out["histogram"] = {"edges": [_plain(e) for e in edges], "counts": counts.tolist()}
            chart = histogram(f"{name}", edges, counts)
    elif out["kind"] == kinds.DATETIME:
        stamps = pd.to_datetime(present, errors="coerce").dropna()
        if len(stamps):
            out.update(first=stamps.min().isoformat(), last=stamps.max().isoformat())
    if out["kind"] != kinds.NUMERIC or out["distinct"] <= FEW_VALUES:
        counts = present.astype(str).value_counts()
        top = counts.head(10)
        out["top"] = [{"value": _label(v, samples, i), "count": int(c),
                       "share": round(float(c / len(series)), 4)} for i, (v, c) in enumerate(top.items())]
        out["other"] = int(counts.iloc[10:].sum())
        chart = bar(f"{name}", [str(v) for v in top.index], top.tolist(), "rows")
    return out, chart


# ── relations ────────────────────────────────────────────────────────────────

def _correlation_ratio(groups: pd.Series, values: pd.Series) -> float:
    """η: how much of a number's spread lies between the groups of a category,
    0 to 1."""
    data = pd.DataFrame({"g": groups.astype(str), "v": values}).dropna()
    if len(data) < 3 or data["v"].std() == 0:
        return 0.0
    mean = data["v"].mean()
    between = data.groupby("g")["v"].agg(lambda v: len(v) * (v.mean() - mean) ** 2).sum()
    total = ((data["v"] - mean) ** 2).sum()
    return float(np.sqrt(between / total)) if total else 0.0


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V between two categories, 0 to 1."""
    table = pd.crosstab(a.astype(str), b.astype(str))
    if min(table.shape) < 2:
        return 0.0
    n = table.values.sum()
    expected = np.outer(table.sum(1), table.sum(0)) / n
    chi2 = float((((table.values - expected) ** 2) / np.where(expected > 0, expected, 1)).sum())
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def association(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    """How strongly two columns go together, 0 to 1, and how it was measured:
    Spearman's ρ (as |ρ|) between numbers, the correlation ratio η between a
    category and numbers, Cramér's V between categories."""
    a_numeric, b_numeric = kind(a) == kinds.NUMERIC, kind(b) == kinds.NUMERIC
    if a_numeric and b_numeric:
        pair = pd.DataFrame({"a": _numbers(a), "b": _numbers(b)}).dropna()
        if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
            return 0.0, "spearman"
        return float(abs(pair["a"].corr(pair["b"], method="spearman"))), "spearman"
    if a_numeric:
        return _correlation_ratio(b, _numbers(a)), "correlation ratio"
    if b_numeric:
        return _correlation_ratio(a, _numbers(b)), "correlation ratio"
    return _cramers_v(a, b), "cramér's v"


def target_relations(frame: pd.DataFrame, top: int = 10) -> tuple[dict, dict]:
    """Each feature's association with the target, strongest first."""
    data = sampled(frame)
    target = data.iloc[:, -1]
    if is_classes(target) and kind(target) == kinds.NUMERIC:
        target = target.astype(str)
    rows = []
    for column in data.columns[:-1]:
        strength, measure = association(data[column], target)
        rows.append({"column": str(column), "strength": round(strength, 4), "measure": measure})
    rows.sort(key=lambda r: -r["strength"])
    rows = rows[:max(1, int(top))]
    out = {"target": target_name(frame), "sampled_rows": int(len(data)), "relations": rows}
    return out, bar(f"Association with {target_name(frame)}",
                    [r["column"] for r in rows], [r["strength"] for r in rows], "strength")


def correlations(frame: pd.DataFrame, names=None, method: str = "spearman",
                 top: int = 10) -> tuple[dict, dict | None]:
    """The most strongly correlated pairs among numeric features."""
    if method not in ("spearman", "pearson"):
        raise DataError("method is 'spearman' or 'pearson'")
    data = sampled(frame)
    chosen = [str(column_of(data, n).name) for n in names] if names else \
        [str(c) for c in data.columns[:-1] if kind(data[c]) == kinds.NUMERIC]
    numeric = pd.DataFrame({c: _numbers(data[c]) for c in chosen})
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return {"pairs": [], "note": "fewer than two numeric columns that vary"}, None
    matrix = numeric.corr(method=method)
    pairs = [{"a": a, "b": b, "correlation": round(float(matrix.loc[a, b]), 4)}
             for i, a in enumerate(matrix.columns) for b in matrix.columns[i + 1:]
             if np.isfinite(matrix.loc[a, b])]
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    out = {"method": method, "sampled_rows": int(len(data)), "pairs": pairs[:max(1, int(top))]}
    return out, heatmap(f"{method.capitalize()} correlation", list(matrix.columns),
                        np.round(matrix.to_numpy(), 3).tolist())


def missing_values(frame: pd.DataFrame) -> tuple[dict, dict | None]:
    """Missing values per column, and which columns go missing together."""
    missing = frame.isna()
    per_column = [{"column": str(c), "missing": int(missing[c].sum()),
                   "share": round(float(missing[c].mean()), 4)}
                  for c in frame.columns if missing[c].any()]
    per_column.sort(key=lambda r: -r["missing"])
    together = []
    names = [r["column"] for r in per_column]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = (column_of(missing, a) & column_of(missing, b)).sum()
            either = (column_of(missing, a) | column_of(missing, b)).sum()
            if either and both / either >= 0.5:
                together.append({"a": a, "b": b, "overlap": round(float(both / either), 4)})
    together.sort(key=lambda r: -r["overlap"])
    out = {"rows": int(len(frame)), "rows_with_any_missing": int(missing.any(axis=1).sum()),
           "columns": per_column, "missing_together": together[:10]}
    chart = bar("Missing values", [r["column"] for r in per_column],
                [r["missing"] for r in per_column], "rows") if per_column else None
    return out, chart


# ── the structured query ─────────────────────────────────────────────────────

def _condition(frame: pd.DataFrame, condition: dict) -> pd.Series:
    if not isinstance(condition, dict):
        raise DataError("each condition is {column, op, value}")
    series = column_of(frame, condition.get("column"))
    op = condition.get("op")
    if op not in OPERATORS:
        raise DataError(f"op is one of {list(OPERATORS)}")
    if op == "is_missing":
        return series.isna()
    if op == "not_missing":
        return series.notna()
    value = condition.get("value")
    column_kind = kind(series)
    if op in ("in", "not_in"):
        if not isinstance(value, list):
            raise DataError(f"{op} takes a list of values")
        if column_kind == kinds.NUMERIC:
            hit = _numbers(series).isin([_to_number(v) for v in value])
        else:
            hit = series.astype(str).isin([str(v) for v in value])
        return hit if op == "in" else ~hit & series.notna()
    if column_kind == kinds.NUMERIC:
        left, right = _numbers(series), _to_number(value)
    elif column_kind == kinds.DATETIME:
        left = pd.to_datetime(series, errors="coerce")
        right = pd.to_datetime(value, errors="coerce")
        if pd.isna(right):
            raise DataError(f"{value!r} is not a date")
    else:
        if op not in ("=", "!="):
            raise DataError(f"{series.name!r} holds categories: compare with = != in not_in")
        left, right = series.astype(str), str(value)
    result = {"=": left == right, "!=": left != right, "<": left < right, "<=": left <= right,
              ">": left > right, ">=": left >= right}[op]
    return result.fillna(False) & series.notna()


def _to_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise DataError(f"{value!r} is not a number")


def _grouping(frame: pd.DataFrame, name) -> pd.Series:
    """A column as groups: categories as they are, numbers with many values
    cut into ranges of equal counts, dates by year when they span several."""
    series = column_of(frame, name)
    column_kind = kind(series)
    if column_kind == kinds.NUMERIC and series.nunique() > FEW_VALUES:
        cut = pd.qcut(_numbers(series), q=BINS, duplicates="drop")
        return cut.astype(str).where(series.notna())
    if column_kind == kinds.DATETIME:
        stamps = pd.to_datetime(series, errors="coerce")
        return stamps.dt.year.astype("Int64").astype(str).where(stamps.notna())
    if column_kind == kinds.NUMERIC:
        return _numbers(series).map(lambda v: f"{v:g}").where(series.notna())
    return series.astype(str).where(series.notna())


def _aggregate(frame: pd.DataFrame, spec: dict, groups, total: int) -> tuple[str, object]:
    if not isinstance(spec, dict) or spec.get("fn") not in AGGREGATES:
        raise DataError(f"each aggregate is {{fn, column}} with fn one of {list(AGGREGATES)}")
    fn, name = spec["fn"], spec.get("column")
    if fn in ("count", "share") and not name:
        label = fn
        values = groups.size() if groups is not None else len(frame)
        if fn == "share":
            values = values / total if total else values
        return label, values
    series = column_of(frame, name)
    label = f"{fn}({name})"
    if fn in ("mean", "median", "min", "max", "sum", "std"):
        if kind(series) != kinds.NUMERIC:
            raise DataError(f"{fn} needs a numeric column; {name!r} is {kind(series)}")
        series = _numbers(series)
    if groups is None:
        value = getattr(series, {"count": "count", "share": "count"}.get(fn, fn))()
        if fn == "share":
            value = value / total if total else value
        return label, value
    values = getattr(series.groupby(groups.keys_for_values), {"share": "count"}.get(fn, fn))()
    if fn == "share":
        values = values / total if total else values
    return label, values


class _Groups:
    """The groups a query's rows fall into, by up to two columns."""

    def __init__(self, keys: list[pd.Series]):
        self.keys_for_values = keys
        self._grouped = pd.Series(1, index=keys[0].index).groupby(keys)

    def size(self):
        return self._grouped.size()


def query_data(frame: pd.DataFrame, where=None, group_by=None, aggregate=None,
               order_by=None, limit: int = 20, samples: int = 0) -> dict:
    """Rows matching *where*, grouped by up to two columns, with *aggregate*
    worked out per group — or, with neither, how many match and (as far as
    `ASSISTANT_SAMPLE_ROWS` allows) a few of them."""
    where = where or []
    if not isinstance(where, list):
        raise DataError("where is a list of conditions")
    mask = pd.Series(True, index=frame.index)
    for condition in where:
        mask &= _condition(frame, condition)
    matched = frame[mask]
    out = {"rows": int(len(frame)), "matched": int(len(matched))}
    group_by = [group_by] if isinstance(group_by, str) else list(group_by or [])
    if len(group_by) > 2:
        raise DataError("group by at most two columns")
    aggregate = aggregate or ([{"fn": "count"}] if group_by else [])
    if not group_by and not aggregate:
        if samples:
            out["examples"] = [{str(k): _plain(v) for k, v in row.items()}
                               for row in matched.head(samples).to_dict("records")]
        return out
    groups = _Groups([_grouping(matched, name) for name in group_by]) if group_by else None
    if groups is None:
        out["values"] = {label: _plain(value)
                         for label, value in (_aggregate(matched, spec, None, len(matched))
                                              for spec in aggregate)}
        return out
    table = pd.DataFrame({label: values for label, values in
                          (_aggregate(matched, spec, groups, len(matched)) for spec in aggregate)})
    if order_by:
        by = order_by.get("by") if isinstance(order_by, dict) else order_by
        if by not in table.columns:
            raise DataError(f"order by one of {list(table.columns)}")
        table = table.sort_values(by, ascending=not (isinstance(order_by, dict)
                                                     and order_by.get("descending")))
    limit = max(1, min(int(limit or 20), MAX_GROUPS))
    out["groups"] = int(len(table))
    out["group_by"] = [str(g) for g in group_by]
    rows = []
    for index, (key, values) in enumerate(table.head(limit).iterrows()):
        key = key if isinstance(key, tuple) else (key,)
        labels = [k if _is_range(k) else _label(k, samples, index) for k in key]
        rows.append({"group": labels, **{c: _counted(c, v) for c, v in values.items()}})
    out["result"] = rows
    return out


def _counted(label: str, value):
    """*value* as JSON has it, a whole number where *label* counts."""
    value = _plain(value)
    if (label == "count" or label.startswith(("count(", "nunique("))) and value is not None:
        return int(value)
    return value


def _is_range(label) -> bool:
    """A numeric range or a year — not a value of the data, so always shown."""
    text = str(label)
    return (text.startswith(("(", "[")) and text.endswith(("]", ")"))) or text.isdigit()


def query_chart(result: dict) -> dict | None:
    """A bar chart of a grouped query's first aggregate."""
    rows = result.get("result") or []
    if not rows:
        return None
    measure = next(k for k in rows[0] if k != "group")
    return bar(measure, [" · ".join(map(str, r["group"])) for r in rows],
               [r[measure] for r in rows], measure)


# ── time ─────────────────────────────────────────────────────────────────────

def time_profile(frame: pd.DataFrame, time_column, series_column=None) -> tuple[dict, dict | None]:
    """How a time-ordered dataset is laid out: its range, its spacing and gaps,
    how many series, the season, and which way the target trends."""
    from core.forecasting import infer_season

    times = column_of(frame, time_column)
    stamps = pd.to_datetime(times, errors="coerce") if kind(times) == kinds.DATETIME else None
    out = {"time_column": str(time_column)}
    if series_column:
        series = column_of(frame, series_column)
        lengths = series.astype(str).value_counts()
        out["series"] = {"count": int(len(lengths)), "shortest": int(lengths.min()),
                         "longest": int(lengths.max())}
    target = _numbers(frame.iloc[:, -1])
    if stamps is not None:
        ordered = pd.DatetimeIndex(np.unique(stamps.dropna().to_numpy()))
        out.update(first=ordered.min().isoformat() if len(ordered) else None,
                   last=ordered.max().isoformat() if len(ordered) else None,
                   distinct_times=int(len(ordered)))
        if len(ordered) >= 3:
            steps = pd.Series(ordered[1:] - ordered[:-1])
            median = steps.median()
            out["step"] = str(median)
            out["gaps"] = int((steps > 1.5 * median).sum())
            out["frequency"] = pd.infer_freq(ordered) if len(ordered) >= 3 else None
            out["season"] = int(infer_season(ordered))
        position = (stamps - stamps.min()).dt.total_seconds()
    else:
        numbers = _numbers(times)
        out.update(first=_plain(numbers.min()), last=_plain(numbers.max()),
                   distinct_times=int(numbers.nunique()))
        position = numbers
    pair = pd.DataFrame({"t": position, "y": target}).dropna()
    chart = None
    if len(pair) >= 3 and pair["t"].nunique() > 1 and pair["y"].std() > 0:
        slope = np.polyfit((pair["t"] - pair["t"].mean()) / pair["t"].std(),
                           (pair["y"] - pair["y"].mean()) / pair["y"].std(), 1)[0]
        out["trend"] = {"direction": "rising" if slope > 0.1 else "falling" if slope < -0.1 else "flat",
                        "standardised_slope": round(float(slope), 3)}
        trail = pd.DataFrame({"t": times, "y": target}).dropna().sort_values("t")
        trail = trail.iloc[:: max(1, len(trail) // 2000)]
        chart = line(f"{target_name(frame)} over {time_column}",
                     [_plain(v) for v in (pd.to_datetime(trail["t"]) if stamps is not None
                                          else trail["t"])], trail["y"].tolist())
    return out, chart


# ── yardsticks ───────────────────────────────────────────────────────────────

def baseline_scores(frame: pd.DataFrame, task: str) -> dict:
    """What a model that knows nothing scores on each metric of *task*, here —
    the score a failed trial is given (`Metric.null_score`: nothing right for
    accuracy, a coin flip for ROC AUC, always the average for an error), worked
    out on every row rather than per fold."""
    from core.metrics import METRICS, metrics_for

    if task not in (tasks.CLASSIFICATION, tasks.REGRESSION):
        raise DataError("task is 'classification' or 'regression'")
    y = frame.iloc[:, -1].dropna().to_numpy()
    scores = {}
    for name in metrics_for(task):
        try:
            scores[name] = _plain(float(METRICS[name].null_score(y)))
        except Exception:  # noqa: BLE001 — a metric with no null score here is left out
            continue
    return {"task": task, "rows": int(len(y)), "scores": scores}


# ── the brief, worked out once ───────────────────────────────────────────────

def brief(frame: pd.DataFrame, samples: int) -> dict:
    """What anyone planning on this dataset should know before asking: its
    size, the target, problems in the data, and what relates to the target."""
    target = frame.iloc[:, -1]
    out = {"rows": int(len(frame)), "columns": int(frame.shape[1] - 1),
           "target": target_name(frame)}
    if is_classes(target):
        counts = target.dropna().astype(str).value_counts()
        out["target_reads_as"] = "classes"
        out["classes"] = int(len(counts))
        if len(counts):
            out["class_balance"] = {"largest_share": round(float(counts.iloc[0] / counts.sum()), 4),
                                    "smallest_share": round(float(counts.iloc[-1] / counts.sum()), 4)}
    else:
        numbers = _numbers(target).dropna()
        out["target_reads_as"] = "a quantity"
        if len(numbers):
            out["target_spread"] = {"mean": _plain(numbers.mean()), "std": _plain(numbers.std()),
                                    "min": _plain(numbers.min()), "max": _plain(numbers.max()),
                                    "skew": _plain(numbers.skew())}
    flags = []
    for column in frame.columns[:-1]:
        series = frame[column]
        present = series.dropna()
        name = str(column)
        if present.nunique() <= 1:
            flags.append({"column": name, "flag": "constant"})
        elif len(present) and present.astype(str).value_counts().iloc[0] / len(present) >= 0.99:
            flags.append({"column": name, "flag": "nearly constant"})
        if present.nunique() == len(series) and len(series) > FEW_VALUES \
                and kind(series) != kinds.NUMERIC:
            flags.append({"column": name, "flag": "an identifier, most likely"})
        if len(series) and series.isna().mean() >= 0.5:
            flags.append({"column": name, "flag": f"{series.isna().mean():.0%} missing"})
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        flags.append({"rows": duplicates, "flag": "duplicate rows"})
    relations, _chart = target_relations(frame, top=5)
    for relation in relations["relations"]:
        if relation["strength"] >= 0.98:
            flags.append({"column": relation["column"],
                          "flag": "almost determines the target: a leak, most likely"})
    out["flags"] = flags
    out["strongest_relations"] = relations["relations"]
    dated = [str(c) for c in frame.columns[:-1] if kind(frame[c]) == kinds.DATETIME]
    out["date_columns"] = dated[:5]
    if dated:
        out["time"], _chart = time_profile(frame, dated[0])
    missing = frame.isna().sum()
    out["missing_cells"] = int(missing.sum())
    return out


# ── charts, as Plotly specs drawn in the person's browser ───────────────────

def _layout(title: str, **extra) -> dict:
    return {"title": {"text": title}, "margin": {"l": 48, "r": 12, "t": 36, "b": 48},
            "height": 260, **extra}


def histogram(title: str, edges, counts) -> dict:
    edges = [float(e) for e in edges]
    centres = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    return {"data": [{"type": "bar", "x": centres, "y": [int(c) for c in counts], "width": widths}],
            "layout": _layout(title, yaxis={"title": {"text": "rows"}})}


def bar(title: str, labels, values, axis: str) -> dict:
    return {"data": [{"type": "bar", "x": [str(v) for v in labels],
                      "y": [_plain(v) for v in values]}],
            "layout": _layout(title, yaxis={"title": {"text": axis}})}


def heatmap(title: str, labels, matrix) -> dict:
    return {"data": [{"type": "heatmap", "x": labels, "y": labels, "z": matrix,
                      "zmin": -1, "zmax": 1, "colorscale": "RdBu"}],
            "layout": _layout(title, height=320)}


def line(title: str, xs, ys) -> dict:
    return {"data": [{"type": "scatter", "mode": "lines", "x": xs, "y": [_plain(v) for v in ys]}],
            "layout": _layout(title)}
