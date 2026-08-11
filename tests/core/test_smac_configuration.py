"""Configuring SMAC, and the bug that made configuring it pointless.

SMAC spends the first part of a run sampling the space — the initial design —
and only then starts choosing from a fitted model. It sizes that design as a
fraction of the budget it was told about, and the budget it was told about used
to be a fixed 100,000, so the fraction never bit: a 30-trial run was 30 trials
of initial design and the model was never asked for anything. Every SMAC run in
this application was quasi-random search.

The first test here is the one that matters. The rest are the settings that
steer a search that now happens.
"""

import pytest

from core.optimizers import SMACOptimizer

pytestmark = pytest.mark.slow

INITIAL = "Initial Design"


def _origins(optimizer, result):
    """Where each configuration came from, counted by kind."""
    from collections import Counter

    return Counter(
        "initial" if v.startswith(INITIAL) else "model"
        for v in optimizer.serialize_result(result)["config_origins"].values())


def _run(optimizer, models, metrics, iris_splits, trials=30, previous=None):
    X_train, X_val, y_train, y_val = iris_splits
    return optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": trials}, previous_result=previous)


# ── the bug ──────────────────────────────────────────────────────────────────

def test_a_run_reaches_the_model_rather_than_sampling_throughout(
        models, metrics, iris_splits):
    """The regression. Every trial coming out of the initial design means the
    surrogate was fitted every iteration and consulted never — Bayesian
    optimization in name only."""
    optimizer = SMACOptimizer()

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits))

    assert counts["model"] > counts["initial"], counts
    assert counts["initial"] > 0, "some exploration is still expected"


def test_exploration_before_modelling_is_what_moves_that_line(
        models, metrics, iris_splits):
    """And it is the knob for it: a bigger share explores longer."""
    little = SMACOptimizer(exploration_ratio=0.1)
    plenty = SMACOptimizer(exploration_ratio=0.6)

    few = _origins(little, _run(little, models, metrics, iris_splits))
    many = _origins(plenty, _run(plenty, models, metrics, iris_splits))

    assert many["initial"] > few["initial"], (few, many)


# ── saying it as a count instead ─────────────────────────────────────────────

def test_a_count_of_exploration_trials_is_taken_exactly(
        models, metrics, iris_splits):
    """The other way to say the same thing. A share scales with whatever budget
    the run turns out to have; a count is for someone who knows their space and
    wants that many samples and no more."""
    optimizer = SMACOptimizer(exploration_trials=7)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=30))

    assert counts["initial"] == 7, counts


def test_the_count_wins_where_both_are_given(models, metrics, iris_splits):
    """`max_ratio` clamps an explicit count as well as supplying the default
    one, so a count under a small share would silently be cut back down to the
    share — the field would look like it worked and would not have."""
    optimizer = SMACOptimizer(exploration_trials=9, exploration_ratio=0.05)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=30))

    assert counts["initial"] == 9, counts


def test_a_count_larger_than_the_budget_is_capped_rather_than_fatal(
        models, metrics, iris_splits):
    """SMAC raises on an initial design that does not fit in the budget. Someone
    typing 500 into a field next to a 10-trial run should get a run that only
    explores, not a run that refuses to start."""
    optimizer = SMACOptimizer(exploration_trials=500)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=10))

    assert counts["initial"] == 10, counts
    assert counts["model"] == 0


def test_the_room_the_default_configuration_takes_is_left_for_it(
        models, metrics, iris_splits):
    """The model's own defaults are an extra configuration on top of the
    sampled ones, and SMAC counts both against the budget."""
    optimizer = SMACOptimizer(exploration_trials=500, use_default_config=True)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=10))

    assert counts["initial"] == 10, counts


# ── the strategies ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["gp", "rf"])
def test_either_strategy_runs_and_searches(strategy, models, metrics, iris_splits):
    optimizer = SMACOptimizer(search_strategy=strategy)

    result = _run(optimizer, models, metrics, iris_splits, trials=20)

    assert len(result.trials) == 20
    assert _origins(optimizer, result)["model"] > 0


@pytest.mark.parametrize("strategy", ["gp", "rf"])
def test_either_strategy_answers_when_asked_how_sure_it_is(strategy, models, metrics,
                                                           iris_splits):
    """Answering at all is the thing being pinned. The random forest refuses the
    standard deviation the Gaussian process gives happily, and the refusal
    lands inside an `except` — so asking the wrong way turned the criterion off
    for that strategy with no error and no output.

    What the answer is worth is a separate matter: see the test below."""
    X_train, X_val, y_train, y_val = iris_splits
    optimizer = SMACOptimizer(search_strategy=strategy)
    answers = []
    real = optimizer._confidence_nothing_better
    optimizer._confidence_nothing_better = (
        lambda *a, _r=real: answers.append(_r(*a)) or answers[-1])

    optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": 20, "incumbent_confidence": 0.999})

    assert answers, "the criterion was never consulted"
    assert all(a is not None for a in answers), strategy
    assert all(0.0 <= a <= 1.0 for a in answers), answers


def test_the_gaussian_process_gives_an_answer_that_actually_varies(
        models, metrics, iris_splits):
    """And the random forest, at these trial counts, does not — its posterior is
    flat enough that the number sits at one half whatever it has seen. Pinned so
    the difference is a known property rather than a surprise: the criterion is
    worth setting under `gp` and close to inert under `rf`."""
    X_train, X_val, y_train, y_val = iris_splits
    seen = {}
    for strategy in ("gp", "rf"):
        optimizer = SMACOptimizer(search_strategy=strategy)
        answers = []
        real = optimizer._confidence_nothing_better
        optimizer._confidence_nothing_better = (
            lambda *a, _r=real, _s=answers: _s.append(_r(*a)) or _s[-1])
        optimizer.optimize(
            models["Random Forest"], X_train, y_train, X_val, y_val,
            metrics=metrics, primary_metric="accuracy", seed=0,
            stopping={"max_trials": 20, "incumbent_confidence": 0.999})
        seen[strategy] = answers

    assert len(set(seen["gp"])) > 1, seen["gp"]


def test_an_unknown_strategy_falls_back_rather_than_failing(models, metrics, iris_splits):
    """A hand-edited .ihpo can say anything. A search that still runs beats one
    that raises out of the worker."""
    optimizer = SMACOptimizer(search_strategy="quantum")

    assert optimizer.get_params()["search_strategy"] == "gp"
    assert len(_run(optimizer, models, metrics, iris_splits, trials=3).trials) == 3


# ── resuming ─────────────────────────────────────────────────────────────────

def test_a_resumed_run_does_not_start_exploring_again(models, metrics, iris_splits):
    """The history now reaches SMAC by replay rather than from its own stored
    directory. If that left the initial design unconsumed, every resume would
    spend its first trials re-sampling a space it had already sampled."""
    optimizer = SMACOptimizer()
    first = _run(optimizer, models, metrics, iris_splits, trials=20)

    second = _run(optimizer, models, metrics, iris_splits, trials=10, previous=first)
    added = second.trials[len(first.trials):]

    assert len(added) == 10
    origins = optimizer.serialize_result(second)["config_origins"]
    later = [origins[str(t.trial)] for t in added if str(t.trial) in origins]
    assert later and not all(o.startswith(INITIAL) for o in later), later


def test_changed_settings_take_effect_without_losing_the_history(
        models, metrics, iris_splits):
    """Settings are editable between runs precisely because this holds: the
    trials were measured the same way whatever the search strategy was."""
    first = _run(SMACOptimizer(), models, metrics, iris_splits, trials=10)

    second = _run(SMACOptimizer(search_strategy="rf"), models, metrics,
                  iris_splits, trials=5, previous=first)

    assert len(second.trials) == 15
    assert [t.trial for t in second.trials] == list(range(1, 16))
    assert [t.config for t in second.trials[:10]] == [t.config for t in first.trials]
