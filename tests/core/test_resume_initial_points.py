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
    return "".join("I" if (kind or "").lower().startswith(INITIAL.lower()) else "m"
                   for _, kind in sorted(origins.items(), key=lambda kv: int(kv[0])))


def _run(models, metrics, iris_splits, trials, previous=None, **settings):
    """One search, as `(origin string, a result ready to resume from)`.

    The result is round-tripped through serialize/deserialize rather than
    returned live, because that is what a resume actually gets: the run engine
    rebuilds from `Experiment.result` every time. It is also what carries each
    trial's origin and the recorded size of the initial design.
    """
    X_train, X_val, y_train, y_val = iris_splits
    optimizer = SMACOptimizer(**settings)
    result = optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": trials}, previous_result=previous)
    stored = optimizer.serialize_result(result)
    return _origins(optimizer, result), optimizer.deserialize_result(stored)


def test_a_resumed_search_collects_the_points_it_had_not_got_to(
        models, metrics, iris_splits):
    """The mechanism: the design is regenerated from the same seed, the replayed
    trials are recognised as already processed, and the rest are yielded."""
    first, partial = _run(models, metrics, iris_splits, 3)
    whole, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert first.count("I") == 1, first
    assert whole.count("I") == 7, whole
    assert whole[:3] == first, "the replayed three come first, as they were"


def test_the_totals_line_up_when_the_rest_of_the_budget_is_asked_for(
        models, metrics, iris_splits):
    """Three then twenty-seven samples as many points as thirty in one go —
    because SMAC is told the same total either way."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight.count("I") == resumed.count("I") == 7


def test_replaying_a_trial_keeps_where_it_came_from(models, metrics, iris_splits):
    """Resuming rebuilds the facade and replays every past trial with `tell`, and
    a told configuration carrying no origin is stamped `"Custom"` by SMAC — which
    used to relabel the whole of the first run as though the model had chosen it.
    The origin each trial recorded when it was proposed is put back before the
    telling, so the record still says which trials were sampled."""
    first, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert first == "Imm"
    assert resumed[:3] == first, "the first run's three trials, still saying so"


@pytest.mark.parametrize("design", ["sobol", "random"])
def test_a_design_whose_points_nest_samples_as_much_in_total(
        design, models, metrics, iris_splits):
    """The invariant that makes stopping early harmless. Both of these draw
    sequentially from a seeded generator, so a bigger design contains the
    smaller one and SMAC skips what it has already evaluated — the totals come
    out the same however the run was chopped up. Only the placement differs."""
    straight, _ = _run(models, metrics, iris_splits, 24, initial_design=design)
    _, partial = _run(models, metrics, iris_splits, 8, initial_design=design)
    resumed, _ = _run(models, metrics, iris_splits, 16, previous=partial,
                      initial_design=design)

    assert resumed.count("I") == straight.count("I"), (straight, resumed)


def test_a_latin_hypercube_keeps_the_design_it_started_with(
        models, metrics, iris_splits):
    """It does not nest — it stratifies each dimension into `n` bins, so asking
    for a different `n` moves every point. Before this was handled, resuming
    sampled a whole fresh design in the middle of a search that had been
    modelling for nine trials. The size SMAC recorded in its scenario is
    honoured instead, so the resumed run adds none."""
    first, partial = _run(models, metrics, iris_splits, 12,
                          initial_design="latin_hypercube")
    resumed, _ = _run(models, metrics, iris_splits, 24, previous=partial,
                      initial_design="latin_hypercube")

    assert first.count("I") == resumed.count("I"), (first, resumed)
    assert "I" not in resumed[12:], "no sampling once the model has taken over"


def test_but_it_is_not_the_same_experiment(models, metrics, iris_splits):
    """Worth pinning as a known property rather than discovered as a surprise.
    The interrupted run sized its own sampling phase for a three-trial budget
    and got one point, so its trials are not the first three of the long run —
    and everything after them differs accordingly."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight != resumed
    assert straight.startswith("IIIIIII"), "sampled first, then modelled"
    assert resumed.startswith("ImmIIIIII"), "one sampled, two modelled, then the rest"


def test_resuming_with_a_smaller_budget_explores_less_than_was_planned(
        models, metrics, iris_splits):
    """The consequence of there being no stored intent. SMAC is told the budget
    is three plus five, so a quarter of it is two — which the three replayed
    trials have nearly covered already. Asking for a quarter and getting an
    eighth is not a bug in the arithmetic; it is that "a quarter" has always
    meant a quarter of what this run knows about."""
    straight, _ = _run(models, metrics, iris_splits, 8)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 5, previous=partial)

    assert resumed.count("I") == 2, resumed
    assert resumed.count("I") == straight.count("I"), (
        "and still a quarter of the eight trials that were actually run")
