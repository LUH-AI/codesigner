"""Dividing the rows in time order instead of at random.

Every validation row has to come after every row its model trained on, by the
column the experiment is ordered by — with a gap between the two when asked
for — and nothing may shuffle or stratify that order away. Exercised on index
arrays, where "time" is the position, and on `tests/fixtures/flats.csv`, which
has a listing date.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import io, provenance
from core.splits import time_cross_validation, time_holdout

FLATS = Path(__file__).parent.parent / "fixtures" / "flats.csv"


def _positions(n):
    return np.arange(n).reshape(-1, 1), np.arange(n)


def test_a_time_holdout_validates_the_latest_rows():
    """What: the last share of the rows is validated, everything before it is
    trained on, and a gap leaves rows between the two out of both.
    How: 20 rows, 20% held out, a gap of 2."""
    splits = time_holdout(*_positions(20), test_size=0.2, gap=2)
    (train, val), = splits.folds

    assert val.tolist() == [16, 17, 18, 19]
    assert train.tolist() == list(range(14))


def test_rolling_origin_folds_always_train_on_the_past():
    """What: each fold validates a later block than the last, trains only on
    rows before it (less the gap), and the earliest rows are never validated.
    How: 30 rows in 4 folds with a gap of 1."""
    splits = time_cross_validation(*_positions(30), folds=4, gap=1)

    assert len(splits.folds) == 4
    for (train, val), (_, later) in zip(splits.folds, splits.folds[1:]):
        assert val.max() < later.min()
    for train, val in splits.folds:
        assert train.max() < val.min() - 1
    validated = np.concatenate([val for _, val in splits.folds])
    assert 0 not in validated


def test_too_few_rows_is_said_plainly():
    """What: a division that leaves nothing to train on refuses with the
    numbers. How: 5 rows, half held out, a gap of 3."""
    with pytest.raises(ValueError, match="nothing to train on"):
        time_holdout(*_positions(5), test_size=0.5, gap=3)


def test_loading_in_order_sorts_by_the_column_and_leaves_out_unplaced_rows(tmp_path):
    """What: the rows come oldest first by a date column, and a row without a
    date is left out. How: a copy of the fixture with its rows reversed and
    two dates blanked, loaded in order of `listed`."""
    frame = pd.read_csv(FLATS).iloc[::-1]
    frame.iloc[[0, 5], frame.columns.get_loc("listed")] = None
    shuffled = tmp_path / "shuffled.csv"
    frame.to_csv(shuffled, index=False)

    X, y = io._load_frame(shuffled, order_by="listed")
    days = X[:, io.encoding.feature_names(io.dataset_encoding(shuffled)).index("listed:days")]

    assert len(X) == len(y) == 238
    assert np.all(np.diff(days) >= 0)


def test_a_number_column_orders_and_a_label_column_does_not():
    """What: rows can be ordered by a number (a year, a period), never by a
    label. How: orders the fixture by `rooms`, then asks for `city`."""
    X, _ = io._load_frame(FLATS, order_by="rooms")
    rooms = X[:, io.encoding.feature_names(io.dataset_encoding(FLATS)).index("rooms")]
    assert np.all(np.diff(rooms) >= 0)

    with pytest.raises(ValueError, match="labels"):
        io._load_frame(FLATS, order_by="city")


def test_the_suggestions_are_the_dates_then_the_numbers():
    """What: the columns a dataset could be ordered by, dates first, never a
    label or the target. How: reads them off the fixture."""
    assert io.orderable_columns(pd.read_csv(FLATS)) == [
        "listed", "first_viewing", "rooms", "area_m2"]


def test_the_file_says_it_was_divided_in_time_order():
    """What: the evaluation record names the column and the gap, and says it
    was not stratified — whatever the target would have allowed. How: builds
    the record for a time-ordered and a random holdout of the same labels."""
    y = np.array(["a", "b"] * 10)
    ordered = provenance.evaluation(0, y, test_size=0.25, time_column="listed", gap=3)
    shuffled = provenance.evaluation(0, y, test_size=0.25)

    assert ordered["time_column"] == "listed" and ordered["gap"] == 3
    assert ordered["stratified"] is False and shuffled["stratified"] is True
    assert shuffled["time_column"] is None and shuffled["gap"] is None
