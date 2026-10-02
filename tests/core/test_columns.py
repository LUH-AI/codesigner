"""What kind each feature column is, and how it becomes numbers.

`core.columns` decides the kind from the values alone; `core.encoding` writes
down what encoding a frame takes and turns any frame with those columns into
floats. Exercised on small hand-made frames and on `tests/fixtures/flats.csv`,
which has every kind and missing values in both a label and a number column.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import encoding
from core.columns import CATEGORICAL, DATETIME, NUMERIC, infer_kinds, kind_of

FLATS = Path(__file__).parent.parent / "fixtures" / "flats.csv"


def test_numbers_text_flags_and_dates_each_get_their_kind():
    """What: the kind of every column of a realistic file.
    How: infers the fixture's kinds from its CSV as pandas reads it."""
    kinds = infer_kinds(pd.read_csv(FLATS).iloc[:, :-1])

    assert kinds == {"city": CATEGORICAL, "rooms": NUMERIC, "area_m2": NUMERIC,
                     "furnished": CATEGORICAL, "listed": DATETIME,
                     "first_viewing": DATETIME}


def test_a_whole_number_column_stays_a_number():
    """What: few distinct integers are not taken for labels — that is a call
    for someone who knows the data. How: a 1–3 rating column."""
    assert kind_of(pd.Series([1, 2, 3, 2, 1])) == NUMERIC


def test_a_true_false_column_is_a_label():
    """What: booleans are categories, not 0/1 quantities. How: a bool Series."""
    assert kind_of(pd.Series([True, False, True])) == CATEGORICAL


def test_a_date_column_survives_the_odd_unreadable_cell():
    """What: one "unknown" among many dates keeps the column a date; a column
    that is mostly not dates is labels. How: 19 dates and one word, then the
    reverse share."""
    dates = [f"2024-01-{d:02d}" for d in range(1, 20)] + ["unknown"]
    assert kind_of(pd.Series(dates)) == DATETIME
    assert kind_of(pd.Series(dates[:5] + ["a", "b", "c", "d", "e"])) == CATEGORICAL


def test_numbers_written_as_text_are_not_dates():
    """What: "2019", "2020" read as years by pandas, but a column of them is not
    a calendar. How: a text column of four-digit numbers."""
    assert kind_of(pd.Series(["2019", "2020", "2021", "2022"])) != DATETIME


def test_an_empty_column_costs_nothing():
    """What: a column with no values at all is numeric, and encodes as NaN.
    How: an all-None object column."""
    frame = pd.DataFrame({"blank": [None, None]})
    assert kind_of(frame["blank"]) == NUMERIC
    assert np.isnan(encoding.encode(encoding.fit(frame), frame)).all()


def test_labels_become_codes_and_the_unknown_becomes_nan():
    """What: a label's code is its place among the sorted labels; a missing or
    never-seen label is NaN. How: fits on three cities, encodes a fourth."""
    frame = pd.DataFrame({"city": ["Hannover", "Berlin", None, "Hannover"]})
    spec = encoding.fit(frame)

    assert spec["columns"][0]["levels"] == ["Berlin", "Hannover"]
    np.testing.assert_array_equal(
        encoding.encode(spec, frame)[:, 0], [1.0, 0.0, np.nan, 1.0])
    unseen = encoding.encode(spec, pd.DataFrame({"city": ["Munich"]}))
    assert np.isnan(unseen[0, 0])


def test_a_date_becomes_its_calendar_and_an_hour_only_when_there_is_one():
    """What: a date column expands into year, month, day, weekday and days since
    1970, plus the hour when any value has a time. How: one date-only column and
    one timestamp column."""
    frame = pd.DataFrame({"day": ["2024-03-01", "2024-03-04"],
                          "at": ["2024-03-01 08:30", "2024-03-04 17:00"]})
    spec = encoding.fit(frame)

    assert encoding.feature_names(spec) == [
        "day:year", "day:month", "day:day", "day:weekday", "day:days",
        "at:year", "at:month", "at:day", "at:weekday", "at:days", "at:hour"]
    X = encoding.encode(spec, frame)
    np.testing.assert_array_equal(X[0, :4], [2024, 3, 1, 4])   # a Friday
    assert X[1, 4] - X[0, 4] == 3
    assert X[:, -1].tolist() == [8.0, 17.0]


def test_columns_are_found_by_name_not_position():
    """What: a frame with its columns in another order, or one of them
    missing, encodes the same as the frame fitted on. How: reorders and drops
    columns of the fixture."""
    frame = pd.read_csv(FLATS).iloc[:, :-1]
    spec = encoding.fit(frame)
    X = encoding.encode(spec, frame)

    shuffled = frame[list(reversed(frame.columns))]
    np.testing.assert_array_equal(encoding.encode(spec, shuffled), X)
    without = encoding.encode(spec, frame.drop(columns=["rooms"]))
    assert np.isnan(without[:, encoding.feature_names(spec).index("rooms")]).all()


def test_the_encoding_is_plain_json_and_kinds_follow_the_features():
    """What: the spec survives JSON unchanged, and there is one kind per encoded
    column — categorical only for a label's code. How: round-trips the
    fixture's spec and lines its names up with its kinds."""
    spec = encoding.fit(pd.read_csv(FLATS).iloc[:, :-1])

    assert json.loads(json.dumps(spec)) == spec
    kinds = dict(zip(encoding.feature_names(spec), encoding.feature_kinds(spec)))
    assert kinds["city"] == kinds["furnished"] == CATEGORICAL
    assert kinds["listed:month"] == kinds["area_m2"] == NUMERIC
    assert not encoding.is_plain(spec)


@pytest.mark.parametrize("name", ["iris.csv", "wine.csv", "diabetes.csv"])
def test_a_dataset_of_numbers_is_left_as_it_was(name):
    """What: every bundled demo is numbers, and encoding changes nothing but the
    dtype. How: compares the encoded features with the raw ones."""
    frame = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / name).iloc[:, :-1]
    spec = encoding.fit(frame)

    assert encoding.is_plain(spec)
    np.testing.assert_array_equal(encoding.encode(spec, frame), frame.to_numpy(dtype=float))


def test_overriding_a_kind_turns_numbers_into_labels():
    """What: a kind someone states wins over the inferred one. How: fits the
    rooms column as categorical."""
    frame = pd.DataFrame({"rooms": [1, 2, 3, 2]})
    spec = encoding.fit(frame, kinds={"rooms": CATEGORICAL})

    assert spec["columns"][0] == {"name": "rooms", "kind": CATEGORICAL, "levels": ["1", "2", "3"]}
    np.testing.assert_array_equal(encoding.encode(spec, frame)[:, 0], [0, 1, 2, 1])


@pytest.mark.parametrize("values", [["08:30", "17:15", "09:00"], ["May", "June", "July"],
                                    ["1/2", "3/4", "1/3"], ["1st", "2nd", "3rd"]])
def test_a_part_of_a_date_is_a_label_not_a_date(values):
    """What: a time of day, a month, a fraction or an ordinal is not read as a
    date — a parser would complete it with today's year and day, so its
    calendar features would change with the day they were computed.
    How: kinds of four such columns, each a label."""
    assert kind_of(pd.Series(values)) == CATEGORICAL


def test_a_label_read_back_as_a_number_keeps_its_code():
    """What: a label column whose labels are digits encodes the same when a
    later file's copy of it is read as floats (1.0 for "1") — what pandas does
    to a numeric column with a gap. How: fits on text labels, encodes floats."""
    spec = encoding.fit(pd.DataFrame({"grade": ["1", "2", "x"]}))
    later = pd.DataFrame({"grade": [1.0, np.nan, 2.0]})

    np.testing.assert_array_equal(encoding.encode(spec, later)[:, 0], [0.0, np.nan, 1.0])
