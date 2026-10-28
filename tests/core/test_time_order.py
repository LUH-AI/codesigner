"""The time-ordered splits keep every validation row later than every training
row, with an optional gap between them. `tests/fixtures/flats.csv` supplies a
real date column.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from core import datasets, io, provenance
from core.splits import time_cross_validation, time_holdout

FLATS = Path(__file__).parent.parent / "fixtures" / "flats.csv"


def _positions(n):
    return np.arange(n).reshape(-1, 1), np.arange(n)


def test_a_time_holdout_validates_the_latest_rows():
    """`time_holdout` validates the last 20%, and the gap rows join neither
    side."""
    splits = time_holdout(*_positions(20), test_size=0.2, gap=2)
    (train, val), = splits.folds

    assert val.tolist() == [16, 17, 18, 19]
    assert train.tolist() == list(range(14))


def test_rolling_origin_folds_always_train_on_the_past():
    """`time_cross_validation` folds move forward, train before the gap, and
    never validate the first rows."""
    splits = time_cross_validation(*_positions(30), folds=4, gap=1)

    assert len(splits.folds) == 4
    for (train, val), (_, later) in zip(splits.folds, splits.folds[1:]):
        assert val.max() < later.min()
    for train, val in splits.folds:
        assert train.max() < val.min() - 1
    validated = np.concatenate([val for _, val in splits.folds])
    assert 0 not in validated


def test_too_few_rows_is_said_plainly():
    with pytest.raises(ValueError, match="nothing to train on"):
        time_holdout(*_positions(5), test_size=0.5, gap=3)


def test_loading_in_order_sorts_by_the_column_and_leaves_out_unplaced_rows(tmp_path):
    """`load_frame` sorts oldest first by `order_by` and drops rows missing a
    date."""
    frame = pd.read_csv(FLATS).iloc[::-1]
    frame.iloc[[0, 5], cast(int, frame.columns.get_loc("listed"))] = None
    shuffled = tmp_path / "shuffled.csv"
    frame.to_csv(shuffled, index=False)

    X, y = datasets.load_frame(shuffled, order_by="listed")
    days = X[:, io.encoding.feature_names(datasets.dataset_encoding(shuffled)).index("listed:days")]

    assert len(X) == len(y) == 238
    assert np.all(np.diff(days) >= 0)


def test_a_number_column_orders_and_a_label_column_does_not():
    X, _ = datasets.load_frame(FLATS, order_by="rooms")
    rooms = X[:, io.encoding.feature_names(datasets.dataset_encoding(FLATS)).index("rooms")]
    assert np.all(np.diff(rooms) >= 0)

    with pytest.raises(ValueError, match="labels"):
        datasets.load_frame(FLATS, order_by="city")


def test_the_suggestions_are_the_dates_then_the_numbers():
    """`orderable_columns` lists dates first, then numbers, never labels or the
    target."""
    assert datasets.orderable_columns(pd.read_csv(FLATS)) == [
        "listed", "first_viewing", "rooms", "area_m2"]


def test_the_file_says_it_was_divided_in_time_order():
    """`provenance.evaluation` records column and gap, and marks a time-ordered
    split unstratified."""
    y = np.array(["a", "b"] * 10)
    ordered = provenance.evaluation(0, y, test_size=0.25, time_column="listed", gap=3)
    shuffled = provenance.evaluation(0, y, test_size=0.25)

    assert ordered["time_column"] == "listed" and ordered["gap"] == 3
    assert ordered["stratified"] is False and shuffled["stratified"] is True
    assert shuffled["time_column"] is None and shuffled["gap"] is None
