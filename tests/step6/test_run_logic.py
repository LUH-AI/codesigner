"""Step 6: the metric-change decision (ui/services/run_logic.py).

Pure logic derived from app/experiment.py's run-pressed block (lines 200-240).
Deciding what a Run should do given the experiment's committed metrics and the
metric the user chose to optimize this run:

  * first run (no metric committed yet) → run, committing the chosen metric as
    both primary and original;
  * a run that would change the optimized metric while it still matches the
    original → warn (needs confirmation);
  * otherwise → run with the current primary metric.

The confirmation resolves to optimizing the chosen ("new") or the original
("old") metric. `apply_metrics` computes the (primary, original) an experiment
should carry after a run optimizing a given metric: primary always becomes the
optimized metric; original is set once, on the first run.
"""

import pytest

from ui.services.run_logic import apply_metrics, decide_run, resolve_metric_change


def test_first_run_commits_the_chosen_metric():
    """With no metric committed yet, a run optimizes the chosen metric."""
    assert decide_run(original_metric=None, primary_metric=None, chosen_metric="f1") == ("first", "f1")


def test_changing_metric_while_consistent_warns():
    """Choosing a different metric while primary still equals original warns."""
    assert decide_run("accuracy", "accuracy", "f1") == ("warn", "f1")


def test_same_metric_runs_without_warning():
    """Choosing the metric already being optimized just runs, no warning."""
    assert decide_run("accuracy", "accuracy", "accuracy") == ("run", "accuracy")


def test_already_inconsistent_never_rewarns():
    """Once primary has diverged from original, further runs use primary with
    no repeated warning (the label already reads 'Inconsistent')."""
    assert decide_run("accuracy", "f1", "precision") == ("run", "f1")
    assert decide_run("accuracy", "f1", "f1") == ("run", "f1")


def test_resolve_metric_change_new_optimizes_chosen():
    """The dialog's 'new' choice optimizes the chosen (candidate) metric."""
    assert resolve_metric_change("new", original_metric="accuracy", chosen_metric="f1") == "f1"


def test_resolve_metric_change_old_optimizes_original():
    """The dialog's 'old' choice keeps optimizing the original metric."""
    assert resolve_metric_change("old", original_metric="accuracy", chosen_metric="f1") == "accuracy"


def test_resolve_metric_change_cancel_is_none():
    """Cancelling the dialog resolves to no run."""
    assert resolve_metric_change("cancel", original_metric="accuracy", chosen_metric="f1") is None


@pytest.mark.parametrize("original,optimize,expected", [
    (None, "f1", ("f1", "f1")),            # first run: primary and original both set
    ("accuracy", "accuracy", ("accuracy", "accuracy")),  # unchanged
    ("accuracy", "f1", ("f1", "accuracy")),  # 'new': primary moves, original stays → inconsistent
])
def test_apply_metrics(original, optimize, expected):
    """apply_metrics sets primary to the optimized metric and pins original on
    the first run only (it is immutable once set)."""
    assert apply_metrics(original_metric=original, optimize_metric=optimize) == expected
