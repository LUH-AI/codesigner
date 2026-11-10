"""`assistant.tools.data` answers questions from the conversation's dataset
through `assistant/data.py`. Each tool except `_baseline_scores` draws its
finding as a chart under the reply when `show` is set.
"""

from __future__ import annotations

import json

from django.conf import settings

from access.policy import VIEW

from ui.permissions import BOTH

from .. import data
from .registry import Refused, ToolResult, ToolSpec


def source_of(context) -> tuple:
    """`source_of` returns the dataset path and target column of the planning
    dataset, else of the experiment."""
    conversation = context.conversation
    path = conversation.dataset_path()
    if path is not None:
        return path, (conversation.plan or {}).get("target_column") or ""
    if conversation.experiment_id is None:
        raise Refused("there is no dataset yet; ask the person to choose one")
    exp = context.experiment(conversation.experiment_id, VIEW, BOTH)
    path = exp.data.dataset_path()
    if path is None or not path.is_file():
        raise Refused("the experiment's dataset is not on this instance")
    return path, exp.data.target_column or ""


def dataset_of(context):
    """`dataset_of` loads the dataset with the target column last. A target
    missing from the file falls back to the last column."""
    path, target = source_of(context)
    try:
        return data.load(path, target)
    except (KeyError, ValueError):
        return data.load(path)


def brief_for(context) -> dict | None:
    """`brief_for` returns `data.brief` for the dataset, cached on the
    conversation until the file or target changes, or None without a dataset."""
    try:
        path, target = source_of(context)
    except Refused:
        return None
    conversation = context.conversation
    key = f"{path}:{path.stat().st_mtime_ns}:{target}"
    kept = conversation.data_brief or {}
    if kept.get("for") == key:
        return kept.get("facts")
    facts = data.brief(dataset_of(context), _samples())
    conversation.data_brief = {"for": key, "facts": facts}
    conversation.save(update_fields=["data_brief"])
    return facts


def _samples() -> int:
    return max(0, int(settings.ASSISTANT_SAMPLE_ROWS))


def _answer(context, found: dict, chart, show: bool, title: str) -> ToolResult:
    if show and chart:
        from ..loop import emit

        emit(context.conversation, "chart", title=title, spec=chart)
    return ToolResult(output=json.dumps(found, default=str)
                      + (" (Drawn for the person under your reply.)" if show and chart else ""))


def _run(analysis):
    try:
        return analysis()
    except data.DataError as error:
        raise Refused(str(error))


def _describe_column(context, column: str, show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found, chart = _run(lambda: data.describe_column(frame, column, _samples()))
    return _answer(context, found, chart, show, str(column))


def _target_relations(context, top: int = 10, show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found, chart = _run(lambda: data.target_relations(frame, top))
    return _answer(context, found, chart, show, f"Association with {data.target_name(frame)}")


def _correlations(context, columns: list | None = None, method: str = "spearman",
                  top: int = 10, show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found, chart = _run(lambda: data.correlations(frame, columns, method, top))
    return _answer(context, found, chart, show, "Correlations")


def _missing_values(context, show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found, chart = _run(lambda: data.missing_values(frame))
    return _answer(context, found, chart, show, "Missing values")


def _query_data(context, where: list | None = None, group_by: list | None = None,
                aggregate: list | None = None, order_by: dict | None = None,
                limit: int = 20, show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found = _run(lambda: data.query_data(frame, where, group_by, aggregate, order_by, limit,
                                         _samples()))
    return _answer(context, found, data.query_chart(found), show, "Query")


def _time_profile(context, time_column: str, series_column: str = "",
                  show: bool = False) -> ToolResult:
    frame = dataset_of(context)
    found, chart = _run(lambda: data.time_profile(frame, time_column, series_column or None))
    return _answer(context, found, chart, show, "Over time")


def _baseline_scores(context, task: str) -> ToolResult:
    frame = dataset_of(context)
    return ToolResult(output=json.dumps(_run(lambda: data.baseline_scores(frame, task))))


_SHOW = {"type": "boolean", "description": "Also draw it as a chart under your reply."}

DESCRIBE_COLUMN = ToolSpec(
    name="describe_column",
    description=("One column of the dataset: its kind, missing and distinct values; for numbers "
                 "the mean, spread, quantiles, outliers and a histogram; for categories the "
                 "commonest values."),
    parameters={"type": "object", "properties": {"column": {"type": "string"}, "show": _SHOW},
                "required": ["column"]},
    handler=_describe_column,
)

TARGET_RELATIONS = ToolSpec(
    name="target_relations",
    description=("How strongly each column goes with the target, strongest first, 0 to 1: "
                 "Spearman's ρ between numbers, the correlation ratio between a category and "
                 "numbers, Cramér's V between categories. Association, not cause."),
    parameters={"type": "object", "properties": {"top": {"type": "integer"}, "show": _SHOW}},
    handler=_target_relations,
)

CORRELATIONS = ToolSpec(
    name="correlations",
    description="The most strongly correlated pairs among the numeric columns.",
    parameters={"type": "object", "properties": {
        "columns": {"type": "array", "items": {"type": "string"}},
        "method": {"type": "string", "enum": ["spearman", "pearson"]},
        "top": {"type": "integer"}, "show": _SHOW}},
    handler=_correlations,
)

MISSING_VALUES = ToolSpec(
    name="missing_values",
    description="Missing values per column, rows with any, and columns that go missing together.",
    parameters={"type": "object", "properties": {"show": _SHOW}},
    handler=_missing_values,
)

QUERY_DATA = ToolSpec(
    name="query_data",
    description=(
        "Count, filter and aggregate the dataset. `where`: conditions {column, op, value} with "
        "op one of = != < <= > >= in not_in is_missing not_missing (in/not_in take a list). "
        "`group_by`: up to two columns; numbers with many values are cut into ranges. "
        "`aggregate`: {fn, column} with fn one of count mean median min max sum std nunique "
        "share (count and share need no column). `order_by`: {by: an aggregate's label such "
        "as \"mean(price)\", descending}. Without group_by or aggregate: how many rows match."),
    parameters={"type": "object", "properties": {
        "where": {"type": "array", "items": {"type": "object", "properties": {
            "column": {"type": "string"}, "op": {"type": "string", "enum": list(data.OPERATORS)},
            "value": {}}, "required": ["column", "op"]}},
        "group_by": {"type": "array", "items": {"type": "string"}},
        "aggregate": {"type": "array", "items": {"type": "object", "properties": {
            "fn": {"type": "string", "enum": list(data.AGGREGATES)}, "column": {"type": "string"}},
            "required": ["fn"]}},
        "order_by": {"type": "object", "properties": {"by": {"type": "string"},
                                                      "descending": {"type": "boolean"}}},
        "limit": {"type": "integer"}, "show": _SHOW}},
    handler=_query_data,
)

TIME_PROFILE = ToolSpec(
    name="time_profile",
    description=("How the data runs in time along a column: its range, spacing, gaps, season, "
                 "number of series (with `series_column`), and which way the target trends."),
    parameters={"type": "object", "properties": {
        "time_column": {"type": "string"}, "series_column": {"type": "string"}, "show": _SHOW},
        "required": ["time_column"]},
    handler=_time_profile,
)

BASELINE_SCORES = ToolSpec(
    name="baseline_scores",
    description=("What a model that knows nothing scores on each metric of a task on this "
                 "dataset — what a failed trial is given: nothing right for accuracy, a coin "
                 "flip for ROC AUC, always the average for an error. The yardstick results "
                 "are measured against."),
    parameters={"type": "object", "properties": {
        "task": {"type": "string", "enum": ["classification", "regression"]}},
        "required": ["task"]},
    handler=_baseline_scores,
)


def data_tools():
    """`data_tools` excludes `profile_dataset`."""
    return [DESCRIBE_COLUMN, TARGET_RELATIONS, CORRELATIONS, MISSING_VALUES, QUERY_DATA,
            TIME_PROFILE, BASELINE_SCORES]
