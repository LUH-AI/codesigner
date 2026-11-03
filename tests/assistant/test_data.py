"""Questions about a dataset answered from the data (assistant/data.py) and the
tools that ask them (assistant/tools/data.py): each analysis on small frames
with known answers, the structured query's checks, what may leave the machine,
the brief worked out once, and the charts drawn for the person.
"""

import json

import numpy as np
import pandas as pd
import pytest

from assistant import data
from assistant.gateway import Context
from assistant.models import AssistantEvent, Conversation
from assistant.turns import tools_for

pytestmark = pytest.mark.django_db


def _frame():
    """Prices of 40 houses: size and rooms go with the price, the colour is
    noise, one column never changes, and the last two rows repeat."""
    rng = np.random.default_rng(0)
    size = np.linspace(50, 200, 40)
    frame = pd.DataFrame({
        "size": size,
        "rooms": (size // 50).astype(int),
        "region": ["north", "south"] * 20,
        "colour": rng.choice(["red", "blue", "green"], 40),
        "constant": 1,
        "price": size * 1000 + rng.normal(0, 5000, 40),
    })
    frame.loc[38] = frame.loc[39]
    return frame


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "houses.csv"
    _frame().to_csv(path, index=False)
    return path


# ── the analyses ─────────────────────────────────────────────────────────────

def test_a_numeric_column_is_described_with_its_spread_and_a_histogram():
    """What: a numeric column's count, mean, quantiles and histogram, and a
    histogram chart.
    How: the size column, 50 to 200 evenly spaced."""
    found, chart = data.describe_column(_frame(), "size", samples=5)

    assert found["kind"] == "numeric" and found["missing"] == 0
    assert found["min"] == 50 and found["max"] == 200
    assert sum(found["histogram"]["counts"]) == 40
    assert chart["data"][0]["type"] == "bar"


def test_categories_are_named_only_as_far_as_samples_allow():
    """What: a category's commonest values are named when values may leave
    the machine, and stand-ins otherwise; the counts are the same.
    How: the region column described with samples 5, then 0."""
    named, _ = data.describe_column(_frame(), "region", samples=5)
    hidden, _ = data.describe_column(_frame(), "region", samples=0)

    assert {t["value"] for t in named["top"]} == {"north", "south"}
    assert {t["value"] for t in hidden["top"]} == {"value 1", "value 2"}
    assert [t["count"] for t in named["top"]] == [t["count"] for t in hidden["top"]]


def test_what_goes_with_the_target_comes_first():
    """What: size, which sets the price, is the strongest relation and colour,
    noise, among the weakest; each is measured to suit the kinds.
    How: target_relations on the houses."""
    found, _ = data.target_relations(_frame())
    by_column = {r["column"]: r for r in found["relations"]}

    assert found["relations"][0]["column"] in ("size", "rooms")
    assert by_column["size"]["strength"] > 0.95 and by_column["size"]["measure"] == "spearman"
    assert by_column["colour"]["measure"] == "correlation ratio"
    assert by_column["colour"]["strength"] < by_column["size"]["strength"]


def test_the_query_filters_groups_and_aggregates():
    """What: houses over 100 m² grouped by region, with their count and mean
    price, largest mean first.
    How: query_data, checked against pandas."""
    frame = _frame()
    found = data.query_data(frame, where=[{"column": "size", "op": ">", "value": 100}],
                            group_by=["region"],
                            aggregate=[{"fn": "count"}, {"fn": "mean", "column": "price"}],
                            order_by={"by": "mean(price)", "descending": True}, samples=5)
    expected = frame[frame["size"] > 100].groupby("region")["price"].mean()

    assert found["matched"] == int((frame["size"] > 100).sum())
    rows = {r["group"][0]: r for r in found["result"]}
    assert rows["north"]["mean(price)"] == pytest.approx(expected["north"], rel=1e-6)
    assert isinstance(rows["north"]["count"], int)
    assert found["result"][0]["mean(price)"] >= found["result"][1]["mean(price)"]


@pytest.mark.parametrize("where, group_by, aggregate, said", [
    ([{"column": "garden", "op": "=", "value": 1}], None, None, "no column 'garden'"),
    ([{"column": "size", "op": "~", "value": 1}], None, None, "op is one of"),
    ([{"column": "region", "op": ">", "value": "n"}], None, None, "holds categories"),
    (None, ["region"], [{"fn": "mean", "column": "colour"}], "needs a numeric column"),
    (None, ["region", "colour", "rooms"], None, "at most two"),
    ([{"column": "size", "op": ">", "value": "big"}], None, None, "is not a number"),
])
def test_a_query_that_cannot_be_asked_says_why(where, group_by, aggregate, said):
    """What: unknown columns, operators the column's kind cannot take, an
    average of categories, too many groupings and a word for a number are
    refused, with the reason.
    How: each asked of the houses."""
    with pytest.raises(data.DataError, match=said):
        data.query_data(_frame(), where=where, group_by=group_by, aggregate=aggregate)


def test_rows_leave_the_machine_only_as_far_as_samples_allow():
    """What: a query with nothing to aggregate answers how many rows match,
    with at most `samples` of them, and none at 0; group names that are data
    are stand-ins at 0, ranges of numbers are not.
    How: the same queries at samples 2 and 0."""
    frame = _frame()
    some = data.query_data(frame, where=[{"column": "region", "op": "=", "value": "north"}],
                           samples=2)
    none = data.query_data(frame, where=[{"column": "region", "op": "=", "value": "north"}],
                           samples=0)
    hidden = data.query_data(frame, group_by=["region"], samples=0)
    ranges = data.query_data(frame, group_by=["size"], samples=0)

    assert some["matched"] == none["matched"] == int((frame["region"] == "north").sum())
    assert len(some["examples"]) == 2 and "examples" not in none
    assert {r["group"][0] for r in hidden["result"]} == {"value 1", "value 2"}
    assert all(r["group"][0].startswith("(") for r in ranges["result"])


def test_missing_values_are_counted_and_paired():
    """What: per column and per row, and two columns missing on the same rows
    are named together.
    How: size and rooms blanked on the same five rows."""
    frame = _frame()
    frame.loc[:4, ["size", "rooms"]] = np.nan
    found, _ = data.missing_values(frame)

    assert found["rows_with_any_missing"] == 5
    assert {c["column"] for c in found["columns"]} == {"size", "rooms"}
    assert found["missing_together"][0]["overlap"] == 1.0


def test_a_monthly_series_reads_as_monthly_with_its_trend_and_gaps():
    """What: the spacing, a gap, the yearly season and a rising target.
    How: 36 months of a growing number with one month missing."""
    months = pd.date_range("2020-01-01", periods=36, freq="MS").delete(10)
    frame = pd.DataFrame({"month": months, "sales": np.arange(35) * 2.0 + 5})
    found, chart = data.time_profile(frame, "month")

    assert found["season"] == 12 and found["gaps"] == 1
    assert found["trend"]["direction"] == "rising"
    assert chart["data"][0]["mode"] == "lines"


def test_the_yardstick_is_what_a_failed_trial_scores():
    """What: baseline scores are the metrics' own null scores: always the
    average gives an RMSE of the targets' spread.
    How: regression on the houses."""
    found = data.baseline_scores(_frame(), "regression")

    assert found["scores"]["rmse"] == pytest.approx(float(np.std(_frame()["price"])), rel=1e-6)


def test_the_brief_flags_what_a_planner_should_know():
    """What: the brief names the constant column, the repeated rows, a column
    that is the target in disguise, and the target as a quantity.
    How: the houses with a leaked copy of the price."""
    frame = _frame()
    frame.insert(5, "price_again", frame["price"] * 2)
    found = data.brief(frame, samples=0)
    flags = {(f.get("column"), f["flag"]) for f in found["flags"]}

    assert ("constant", "constant") in flags
    assert (None, "duplicate rows") in flags
    assert ("price_again", "almost determines the target: a leak, most likely") in flags
    assert found["target_reads_as"] == "a quantity"


def test_a_changed_file_is_read_again(tmp_path):
    """What: frames are kept between questions, but a file written since is
    read afresh.
    How: one file loaded, rewritten with another row, loaded again."""
    path = tmp_path / "rows.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(path, index=False)
    first = data.load(path)
    pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}).to_csv(path, index=False)
    import os
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10_000_000))

    assert data.load(path) is not first and len(data.load(path)) == 3


# ── the tools ────────────────────────────────────────────────────────────────

def _run(conversation, tool, **args):
    return tools_for(conversation).execute(tool, args, Context(conversation))


def test_the_tools_answer_from_the_conversations_dataset(csv, monkeypatch):
    """What: a planning conversation's tools read its dataset; a query comes
    back as numbers, and with show a chart is sent to the chat.
    How: a conversation holding the houses, a grouped query with show."""
    monkeypatch.setattr(Conversation, "dataset_path", lambda self: csv)
    conversation = Conversation.objects.create()
    result = _run(conversation, "query_data", group_by=["region"],
                  aggregate=[{"fn": "mean", "column": "price"}], show=True)
    found = json.loads(result.output.split(" (Drawn")[0])

    assert found["groups"] == 2
    chart = AssistantEvent.objects.get(kind="chart")
    assert chart.payload["spec"]["data"][0]["type"] == "bar"


def test_a_question_the_tool_cannot_take_is_refused_with_the_reason(csv, monkeypatch):
    """What: an unknown column comes back to the model as a refusal naming the
    columns there are.
    How: describe_column on a column the houses lack."""
    monkeypatch.setattr(Conversation, "dataset_path", lambda self: csv)
    result = _run(Conversation.objects.create(), "describe_column", column="garden")

    assert "Refused" in result.output and "size" in result.output


def test_without_a_dataset_the_tools_say_so():
    """What: a conversation with no dataset and no experiment is told to ask
    for one.
    How: target_relations on an empty conversation."""
    result = _run(Conversation.objects.create(), "target_relations")

    assert "no dataset yet" in result.output


def test_the_brief_is_in_the_system_prompt_and_worked_out_once(csv, monkeypatch):
    """What: a planning conversation's system prompt carries the dataset's
    brief, worked out the first time and kept until the file changes.
    How: the prompt built twice, counting how often the brief is computed."""
    from assistant.prompts import system_prompt

    monkeypatch.setattr(Conversation, "dataset_path", lambda self: csv)
    calls = []
    real = data.brief
    monkeypatch.setattr(data, "brief", lambda frame, samples: calls.append(1) or real(frame, samples))
    conversation = Conversation.objects.create()
    first = system_prompt(conversation)
    second = system_prompt(Conversation.objects.get(pk=conversation.pk))

    assert "What the dataset says" in first and '"constant"' in first
    assert first == second and len(calls) == 1
