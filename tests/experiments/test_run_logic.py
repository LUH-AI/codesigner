"""`decide_run` warns on every change of the optimized metric.
"""

import pytest

from experiments.services.run_logic import apply_metrics, decide_run, resolve_metric_change


def test_first_run_commits_the_chosen_metric():
    assert decide_run(original_metric=None, current_metric=None, chosen_metric="f1") == ("first", "f1")


def test_changing_metric_warns():
    assert decide_run("accuracy", "accuracy", "f1") == ("warn", "f1")


def test_same_metric_runs_without_warning():
    assert decide_run("accuracy", "accuracy", "accuracy") == ("run", "accuracy")


def test_a_second_change_warns_too_rather_than_being_discarded():
    """`decide_run` warns on every change, including the second."""
    assert decide_run("accuracy", "f1", "precision") == ("warn", "precision")


def test_choosing_the_current_metric_never_warns():
    """Even after primary diverged from original."""
    assert decide_run("accuracy", "f1", "f1") == ("run", "f1")


def test_resolve_metric_change_new_optimizes_chosen():
    assert resolve_metric_change("new", current_metric="accuracy", chosen_metric="f1") == "f1"


def test_resolve_metric_change_old_stays_on_the_current_metric():
    assert resolve_metric_change("old", current_metric="f1", chosen_metric="precision") == "f1"


def test_resolve_metric_change_cancel_is_none():
    assert resolve_metric_change("cancel", current_metric="accuracy", chosen_metric="f1") is None


@pytest.mark.parametrize("original,optimize,expected", [
    (None, "f1", ("f1", "f1")),            # first run sets both
    ("accuracy", "accuracy", ("accuracy", "accuracy")),
    ("accuracy", "f1", ("f1", "accuracy")),  # primary moves, original stays
])
def test_apply_metrics(original, optimize, expected):
    """`apply_metrics` pins original on the first run only."""
    assert apply_metrics(original_metric=original, optimize_metric=optimize) == expected
