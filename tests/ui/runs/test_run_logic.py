"""The metric-change decision (ui/services/run_logic.py).

Pure logic derived from app/experiment.py's run-pressed block (lines 200-240).
Deciding what a Run should do given the experiment's committed metrics and the
metric the user chose to optimize this run:

  * first run (no metric committed yet) → run, committing the chosen metric as
    both primary and original;
  * a run that would change the optimized metric → warn (needs confirmation),
    every time and not only the first;
  * otherwise → run with the current primary metric.

The confirmation resolves to optimizing the chosen ("new") or staying on the
current ("old") metric. `apply_metrics` computes the (primary, original) an experiment
should carry after a run optimizing a given metric: primary always becomes the
optimized metric; original is set once, on the first run.
"""

import pytest

from ui.services.run_logic import apply_metrics, decide_run, resolve_metric_change


def test_first_run_commits_the_chosen_metric():
    """With no metric committed yet, a run optimizes the chosen metric."""
    assert decide_run(original_metric=None, primary_metric=None, chosen_metric="f1") == ("first", "f1")


def test_changing_metric_warns():
    """Choosing a different metric than the one being optimized warns."""
    assert decide_run("accuracy", "accuracy", "f1") == ("warn", "f1")


def test_same_metric_runs_without_warning():
    """Choosing the metric already being optimized just runs, no warning."""
    assert decide_run("accuracy", "accuracy", "accuracy") == ("run", "accuracy")


def test_a_second_change_warns_too_rather_than_being_discarded():
    """An experiment that already diverged once used to run with its current
    metric whatever the form asked for, so a second change was dropped without
    telling anyone. Every change is a choice, and gets confirmed."""
    assert decide_run("accuracy", "f1", "precision") == ("warn", "precision")


def test_choosing_the_current_metric_never_warns():
    """Even for an experiment whose primary has diverged from its original."""
    assert decide_run("accuracy", "f1", "f1") == ("run", "f1")


def test_resolve_metric_change_new_optimizes_chosen():
    """The dialog's 'new' choice optimizes the chosen (candidate) metric."""
    assert resolve_metric_change("new", current_metric="accuracy", chosen_metric="f1") == "f1"


def test_resolve_metric_change_old_stays_on_the_current_metric():
    """'Keep' means keep what the experiment is optimizing now. Returning to the
    *original* would be a third metric the user never asked for, on an
    experiment that had already been changed once."""
    assert resolve_metric_change("old", current_metric="f1", chosen_metric="precision") == "f1"


def test_resolve_metric_change_cancel_is_none():
    """Cancelling the dialog resolves to no run."""
    assert resolve_metric_change("cancel", current_metric="accuracy", chosen_metric="f1") is None


@pytest.mark.parametrize("original,optimize,expected", [
    (None, "f1", ("f1", "f1")),            # first run: primary and original both set
    ("accuracy", "accuracy", ("accuracy", "accuracy")),  # unchanged
    ("accuracy", "f1", ("f1", "accuracy")),  # 'new': primary moves, original stays → inconsistent
])
def test_apply_metrics(original, optimize, expected):
    """apply_metrics sets primary to the optimized metric and pins original on
    the first run only (it is immutable once set)."""
    assert apply_metrics(original_metric=original, optimize_metric=optimize) == expected
