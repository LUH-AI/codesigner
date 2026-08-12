"""What stopping early and resuming does to the sampling phase.

A search samples before it models, and a run that is stopped part-way through
that phase leaves it unfinished. Two questions follow, and the answers are not
the same: *does the search go back and collect the rest*, and *is a stopped-then-
resumed experiment the same experiment as one that ran straight through*.

Yes to the first. SMAC skips initial-design configurations already in its
runhistory, and every past trial is replayed into a rebuilt facade before
anything is asked of it, so the ones already evaluated are recognised and the
rest are collected.

No to the second, and the reason is worth having written down: the size of the
initial design is worked out from *the budget this run was told about*, which is
the trials already done plus this run's cap. There is no stored intent — nothing
anywhere says "this experiment was meant to be thirty trials" — so resuming with
a different cap sizes the sampling phase differently. Resume with the rest of
the original budget and the totals line up; resume with less and the experiment
ends up having explored less than the original plan called for.

These are slow because only a real search produces the config origins that say
where each trial came from.
"""

import pytest

from core.optimizers import SMACOptimizer

pytestmark = pytest.mark.slow

INITIAL = "Initial Design"


def _origins(optimizer, result):
    """One character per trial, in order: I sampled, m chosen by the model."""
    origins = optimizer.serialize_result(result)["config_origins"]
    return "".join("I" if kind.startswith(INITIAL) else "m"
                   for _, kind in sorted(origins.items(), key=lambda kv: int(kv[0])))


def _run(models, metrics, iris_splits, trials, previous=None):
    X_train, X_val, y_train, y_val = iris_splits
    optimizer = SMACOptimizer()
    result = optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": trials}, previous_result=previous)
    return _origins(optimizer, result), result


def test_a_resumed_search_collects_the_points_it_had_not_got_to(
        models, metrics, iris_splits):
    """The mechanism: the design is regenerated from the same seed, the replayed
    trials are recognised as already processed, and the rest are yielded."""
    first, partial = _run(models, metrics, iris_splits, 3)
    whole, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert first.count("I") == 1, first
    assert whole.count("I") == 6, whole
    assert whole.startswith("mmm"), "the replayed three come first"


def test_the_totals_line_up_when_the_rest_of_the_budget_is_asked_for(
        models, metrics, iris_splits):
    """Three then twenty-seven samples as many points as thirty in one go —
    because SMAC is told the same total either way.

    Counted across both runs rather than off the resumed one, for the reason the
    next test is about: the replayed trials come back labelled as though the
    model had chosen them."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    first, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight.count("I") == 7
    assert first.count("I") + resumed.count("I") == 7


def test_replaying_a_trial_loses_where_it_came_from(models, metrics, iris_splits):
    """A known gap in the record, pinned rather than left to be discovered.

    Resuming rebuilds the facade and replays every past trial into it with
    `tell`, which is what lets a changed metric or a lost run directory be
    recovered from. But a told trial carries no origin, so SMAC's runhistory —
    which the exported `.ihpo` copies verbatim — records the replayed ones as
    though the model had chosen them. The trials, their configurations and their
    scores are all intact; only the label for how each was arrived at is not.

    Nothing downstream reads `config_origins`, so this costs nothing today. It
    costs a reader of the file, which is the point of the file."""
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert resumed[:3] == "mmm", "the first run sampled one of these"


def test_but_it_is_not_the_same_experiment(models, metrics, iris_splits):
    """Worth pinning as a known property rather than discovered as a surprise.
    The interrupted run sized its own sampling phase for a three-trial budget
    and got one point, so its trials are not the first three of the long run —
    and everything after them differs accordingly."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight != resumed
    assert straight.startswith("IIIIIII")
    assert resumed.startswith("mmmIIIIII")


def test_resuming_with_a_smaller_budget_explores_less_than_was_planned(
        models, metrics, iris_splits):
    """The consequence of there being no stored intent. SMAC is told the budget
    is three plus five, so a quarter of it is two — which the three replayed
    trials have nearly covered already. Asking for a quarter and getting an
    eighth is not a bug in the arithmetic; it is that "a quarter" has always
    meant a quarter of what this run knows about."""
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 5, previous=partial)

    assert resumed.count("I") == 1, resumed
