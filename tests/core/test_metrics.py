"""Scoring is the application's job.

A model returns predictions and never sees the validation labels, so these are
the only place a score is computed — which is what makes two models comparable
even when they run in different environments with different library versions.
"""

import numpy as np
import pytest

from core.metrics import METRICS, resolve, score_all


def test_the_four_metric_names_are_the_stored_ones():
    """The names go into .ihpo files verbatim, so renaming one breaks every
    experiment that used it."""
    assert set(METRICS) == {"accuracy", "f1", "precision", "recall(macro)"}


def test_a_perfect_prediction_scores_one():
    y = np.array(["a", "b", "a", "b"], dtype=object)
    assert score_all(y, list(y), METRICS) == {name: 1.0 for name in METRICS}


def test_scores_are_plain_floats():
    """sklearn returns numpy scalars, which used to travel all the way into
    TrialResult.scores and rely on the serializer to coerce them."""
    y = np.array([0, 1, 0, 1])
    for value in score_all(y, [0, 1, 1, 1], METRICS).values():
        assert type(value) is float


def test_predictions_may_be_a_plain_list_against_object_labels():
    """That is how they come back from a model in another process."""
    y = np.array(["setosa", "virginica", "setosa"], dtype=object)
    assert score_all(y, ["setosa", "setosa", "setosa"], METRICS)["accuracy"] == pytest.approx(2 / 3)


def test_resolve_picks_named_metrics():
    picked = resolve(["accuracy", "f1"])
    assert set(picked) == {"accuracy", "f1"}


def test_resolve_names_the_ones_that_do_not_exist():
    with pytest.raises(ValueError, match="nope"):
        resolve(["accuracy", "nope"])
