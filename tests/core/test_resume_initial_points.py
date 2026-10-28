"""A resumed SMAC run collects the initial-design points its first run missed,
since SMAC skips configurations already in the replayed runhistory. SMAC sizes
the design from the trials done plus the current cap, and nothing stores a
planned total, so a resume with a smaller cap samples less. `pytestmark` marks
the module slow because only a real search records config origins.
"""

import pytest

from core.optimizers import SMACOptimizer

pytestmark = pytest.mark.slow

INITIAL = "Initial Design"


def _origins(optimizer, result):
    """`_origins` marks each trial `I` (sampled) or `m` (modelled)."""
    origins = optimizer.serialize_result(result)["config_origins"]
    return "".join("I" if (kind or "").lower().startswith(INITIAL.lower()) else "m"
                   for _, kind in sorted(origins.items(), key=lambda kv: int(kv[0])))


def _run(models, metrics, iris_splits, trials, previous=None, **settings):
    """`_run` returns `(origins, result)`, the result round-tripped as a resume
    reads it from `Experiment.result`.
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
    """SMAC regenerates the design from the seed and skips the replayed trials."""
    first, partial = _run(models, metrics, iris_splits, 3)
    whole, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert first.count("I") == 1, first
    assert whole.count("I") == 7, whole
    assert whole[:3] == first, "the replayed three come first, as they were"


def test_the_totals_line_up_when_the_rest_of_the_budget_is_asked_for(
        models, metrics, iris_splits):
    """SMAC is told thirty in total either way."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight.count("I") == resumed.count("I") == 7


def test_replaying_a_trial_keeps_where_it_came_from(models, metrics, iris_splits):
    """`SMACOptimizer` restores the recorded origin of each trial before `tell`,
    because SMAC stamps a told configuration without one `"Custom"`."""
    first, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert first == "Imm"
    assert resumed[:3] == first, "the first run's three trials, still saying so"


@pytest.mark.parametrize("design", ["sobol", "random"])
def test_a_design_whose_points_nest_samples_as_much_in_total(
        design, models, metrics, iris_splits):
    """A bigger Sobol or random design contains the smaller one, because both
    draw sequentially from a seeded generator."""
    straight, _ = _run(models, metrics, iris_splits, 24, initial_design=design)
    _, partial = _run(models, metrics, iris_splits, 8, initial_design=design)
    resumed, _ = _run(models, metrics, iris_splits, 16, previous=partial,
                      initial_design=design)

    assert resumed.count("I") == straight.count("I"), (straight, resumed)


def test_a_latin_hypercube_keeps_the_design_it_started_with(
        models, metrics, iris_splits):
    """A new `n` shifts every point of a Latin hypercube, and a resume keeps the
    size SMAC recorded in its scenario."""
    first, partial = _run(models, metrics, iris_splits, 12,
                          initial_design="latin_hypercube")
    resumed, _ = _run(models, metrics, iris_splits, 24, previous=partial,
                      initial_design="latin_hypercube")

    assert first.count("I") == resumed.count("I"), (first, resumed)
    assert "I" not in resumed[12:], "no sampling once the model has taken over"


def test_but_it_is_not_the_same_experiment(models, metrics, iris_splits):
    """The interrupted run sized its design for three trials and got one point,
    and its trials differ from the long run."""
    straight, _ = _run(models, metrics, iris_splits, 30)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 27, previous=partial)

    assert straight != resumed
    assert straight.startswith("IIIIIII"), "sampled first, then modelled"
    assert resumed.startswith("ImmIIIIII"), "one sampled, two modelled, then the rest"


def test_resuming_with_a_smaller_budget_explores_less_than_was_planned(
        models, metrics, iris_splits):
    """SMAC sizes the design as a quarter of three plus five trials, and the
    replayed trials already hold one of the two design points."""
    straight, _ = _run(models, metrics, iris_splits, 8)
    _, partial = _run(models, metrics, iris_splits, 3)
    resumed, _ = _run(models, metrics, iris_splits, 5, previous=partial)

    assert resumed.count("I") == 2, resumed
    assert resumed.count("I") == straight.count("I"), (
        "and still a quarter of the eight trials that were actually run")
