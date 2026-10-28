"""`core.columns` infers each column's kind from its values, and `core.encoding`
records an encoding and turns matching frames into floats.
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
    kinds = infer_kinds(pd.read_csv(FLATS).iloc[:, :-1])

    assert kinds == {"city": CATEGORICAL, "rooms": NUMERIC, "area_m2": NUMERIC,
                     "furnished": CATEGORICAL, "listed": DATETIME,
                     "first_viewing": DATETIME}


def test_a_whole_number_column_stays_a_number():
    """Few distinct integers stay numeric; labelling them is the data owner's
    call."""
    assert kind_of(pd.Series([1, 2, 3, 2, 1])) == NUMERIC


def test_a_true_false_column_is_a_label():
    """Booleans are categories rather than 0/1 quantities."""
    assert kind_of(pd.Series([True, False, True])) == CATEGORICAL


def test_a_date_column_survives_the_odd_unreadable_cell():
    """A mostly-date column tolerates one unreadable cell."""
    dates = [f"2024-01-{d:02d}" for d in range(1, 20)] + ["unknown"]
    assert kind_of(pd.Series(dates)) == DATETIME
    assert kind_of(pd.Series(dates[:5] + ["a", "b", "c", "d", "e"])) == CATEGORICAL


def test_numbers_written_as_text_are_not_dates():
    """Pandas reads "2019" as a year, yet such a column is no calendar."""
    assert kind_of(pd.Series(["2019", "2020", "2021", "2022"])) != DATETIME


def test_an_empty_column_costs_nothing():
    """`encoding` reads an all-None column as numeric NaN values."""
    frame = pd.DataFrame({"blank": [None, None]})
    assert kind_of(frame["blank"]) == NUMERIC
    assert np.isnan(encoding.encode(encoding.fit(frame), frame)).all()


def test_labels_become_codes_and_the_unknown_becomes_nan():
    """Codes follow sorted label order, and NaN marks a missing or unseen label."""
    frame = pd.DataFrame({"city": ["Hannover", "Berlin", None, "Hannover"]})
    spec = encoding.fit(frame)

    assert spec["columns"][0]["levels"] == ["Berlin", "Hannover"]
    np.testing.assert_array_equal(
        encoding.encode(spec, frame)[:, 0], [1.0, 0.0, np.nan, 1.0])
    unseen = encoding.encode(spec, pd.DataFrame({"city": ["Munich"]}))
    assert np.isnan(unseen[0, 0])


def test_a_date_becomes_its_calendar_and_an_hour_only_when_there_is_one():
    """The hour feature appears only when some value has a time."""
    frame = pd.DataFrame({"day": ["2024-03-01", "2024-03-04"],
                          "at": ["2024-03-01 08:30", "2024-03-04 17:00"]})
    spec = encoding.fit(frame)

    assert encoding.feature_names(spec) == [
        "day:year", "day:month", "day:day", "day:weekday", "day:days",
        "at:year", "at:month", "at:day", "at:weekday", "at:days", "at:hour"]
    X = encoding.encode(spec, frame)
    np.testing.assert_array_equal(X[0, :4], [2024, 3, 1, 4])   # Friday
    assert X[1, 4] - X[0, 4] == 3
    assert X[:, -1].tolist() == [8.0, 17.0]


def test_columns_are_found_by_name_not_position():
    """`encoding` fills a missing column with NaN values."""
    frame = pd.read_csv(FLATS).iloc[:, :-1]
    spec = encoding.fit(frame)
    X = encoding.encode(spec, frame)

    shuffled = frame[list(reversed(frame.columns))]
    np.testing.assert_array_equal(encoding.encode(spec, shuffled), X)
    without = encoding.encode(spec, frame.drop(columns=["rooms"]))
    assert np.isnan(without[:, encoding.feature_names(spec).index("rooms")]).all()


def test_the_encoding_is_plain_json_and_kinds_follow_the_features():
    """Only a label's code is categorical among the encoded features."""
    spec = encoding.fit(pd.read_csv(FLATS).iloc[:, :-1])

    assert json.loads(json.dumps(spec)) == spec
    kinds = dict(zip(encoding.feature_names(spec), encoding.feature_kinds(spec)))
    assert kinds["city"] == kinds["furnished"] == CATEGORICAL
    assert kinds["listed:month"] == kinds["area_m2"] == NUMERIC
    assert not encoding.is_plain(spec)


@pytest.mark.parametrize("name", ["iris.csv", "wine.csv", "diabetes.csv"])
def test_a_dataset_of_numbers_is_left_as_it_was(name):
    frame = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / name).iloc[:, :-1]
    spec = encoding.fit(frame)

    assert encoding.is_plain(spec)
    np.testing.assert_array_equal(encoding.encode(spec, frame), frame.to_numpy(dtype=float))


def test_overriding_a_kind_turns_numbers_into_labels():
    frame = pd.DataFrame({"rooms": [1, 2, 3, 2]})
    spec = encoding.fit(frame, kinds={"rooms": CATEGORICAL})

    assert spec["columns"][0] == {"name": "rooms", "kind": CATEGORICAL, "levels": ["1", "2", "3"]}
    np.testing.assert_array_equal(encoding.encode(spec, frame)[:, 0], [0, 1, 2, 1])


@pytest.mark.parametrize("values", [["08:30", "17:15", "09:00"], ["May", "June", "July"],
                                    ["1/2", "3/4", "1/3"], ["1st", "2nd", "3rd"]])
def test_a_part_of_a_date_is_a_label_not_a_date(values):
    """A parser would fill in today's year and day, making calendar features
    drift with the date."""
    assert kind_of(pd.Series(values)) == CATEGORICAL


def test_a_label_read_back_as_a_number_keeps_its_code():
    """Pandas reads a digit label column with a gap as floats, and the codes
    must survive that."""
    spec = encoding.fit(pd.DataFrame({"grade": ["1", "2", "x"]}))
    later = pd.DataFrame({"grade": [1.0, np.nan, 2.0]})

    np.testing.assert_array_equal(encoding.encode(spec, later)[:, 0], [0.0, np.nan, 1.0])
