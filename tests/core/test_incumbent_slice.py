"""Dragging the figure's curve states a prior, a density on ConfigSpace's
normalized representation. Each grid value must carry the normalized coordinate
SMAC evaluates a prior at.
"""

import math
import random
from typing import cast

import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from core import analytics
from core.analytics import fit_surrogate, pair_trials_with_scores
from core.analytics.surrogate import predict_with_spread
from core.optimizers import RandomOptimizer
from core.optimizers.base import TrialResult


@pytest.fixture
def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([
        Float("lr", (1e-4, 1e-1), log=True),   # raw vs normalized
        Integer("depth", (1, 20)),
        Categorical("kern", ["rbf", "linear", "poly"]),
    ])
    return cs


@pytest.fixture
def trials(space):
    """`trials` contains thirty trials peaking in `lr` near 1e-2."""
    rng = random.Random(7)
    out = []
    for i, config in enumerate(space.sample_configuration(30)):
        values = dict(config)
        score = (0.55
                 + 0.35 * math.exp(-((math.log10(values["lr"]) + 2.0) / 0.6) ** 2)
                 - 0.004 * values["depth"]
                 + rng.gauss(0, 0.01))
        out.append(TrialResult(
            trial=i, config=values, scores={"accuracy": score}, score=score,
            incumbent_score=score, incumbent_config=values,
            run_info={"time": 0.2 + 3.0 * values["depth"] / 20.0}))
    return out


@pytest.fixture
def optimizer():
    return RandomOptimizer()


def test_a_log_hyperparameters_positions_are_evenly_spaced(optimizer, space, trials):
    """Positions spread evenly across [0, 1] while `lr` values stay log-spaced;
    the axis midpoint is lr ≈ 3e-3.
    """
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=5)
    values, positions, warning = sl.grid, sl.positions, sl.warning

    assert warning is None
    assert positions == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert values[0] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-1)
    # The middle position is the geometric mean.
    assert values[2] == pytest.approx(math.sqrt(1e-4 * 1e-1))


def test_an_integer_hyperparameters_positions_are_not_evenly_spaced(optimizer, space, trials):
    """Positions come per value because `_hp_grid` collapses an integer grid to
    its distinct values.
    """
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "depth", n_points=9)
    values, positions, mu = sl.grid, sl.positions, sl.mu

    assert values == sorted(set(values)), "distinct and ordered"
    assert len(positions) == len(values) == len(mu)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert max(gaps) - min(gaps) > 1e-6, "not evenly spaced, and must not be treated as if it were"


def test_a_categoricals_positions_are_its_choice_indices(optimizer, space, trials):
    """ConfigSpace encodes a categorical as the choice index, and the caller
    then draws bars."""
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "kern")
    values, positions = sl.grid, sl.positions

    assert values == ["rbf", "linear", "poly"]
    assert positions == [0.0, 1.0, 2.0]


def test_every_returned_list_has_the_same_length(optimizer, space, trials):
    for name in ("lr", "depth", "kern"):
        sl = analytics.compute_incumbent_slice(
            optimizer, space, trials, "accuracy", name, n_points=17)
        values, positions, mu, sigma = sl.grid, sl.positions, sl.mu, sl.sigma
        assert len(values) == len(positions) == len(mu) == len(sigma) > 1, name


def test_the_spread_is_never_negative(optimizer, space, trials):
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=31)
    sigma = sl.sigma

    assert all(s >= 0 for s in sigma)


def test_the_slice_finds_the_peak_it_was_given(optimizer, space, trials):
    """The slice shows the peak near lr = 1e-2."""
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=61)
    values, mu = sl.grid, sl.mu

    peak = values[mu.index(max(mu))]
    assert 1e-3 < peak < 1e-1


def test_a_second_slice_cuts_through_the_same_configuration(optimizer, space, trials):
    """The metric picks the incumbent for both panels because the constraint
    panel is read against the objective panel."""
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=21)
    a_values, a_positions, a_mu = sl.grid, sl.positions, sl.mu
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=21, values=lambda t: t.duration)
    b_values, b_positions, b_mu = sl.grid, sl.positions, sl.mu

    assert a_values == b_values
    assert a_positions == b_positions
    assert a_mu != b_mu, "a different quantity, or the values= argument did nothing"


def test_duration_is_modellable_although_no_metric_reports_it(optimizer, space, trials):
    """`duration` is a `TrialResult` property, hence the callable `values=`."""
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "depth", n_points=9, values=lambda t: t.duration)
    mu, warning = sl.mu, sl.warning

    assert warning is None
    assert all(v > 0 for v in mu), "seconds, so positive"


def test_an_unknown_hyperparameter_is_a_warning(optimizer, space, trials):
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "nonesuch")
    values, positions, warning = sl.grid, sl.positions, sl.warning

    assert (values, positions) == ([], [])
    assert warning and "nonesuch" in warning


def test_too_few_trials_is_a_warning(optimizer, space, trials):
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials[:1], "accuracy", "lr")
    values, warning = sl.grid, sl.warning

    assert values == []
    assert warning


def test_a_metric_no_trial_carries_is_a_warning(optimizer, space, trials):
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "f1", "lr")
    values, warning = sl.grid, sl.warning

    assert values == []
    assert warning


def test_the_mean_is_the_forests_own_prediction(space, trials):
    """`predict_with_spread` takes the mean from the per-tree stack, and that
    mean must equal `rf.predict`."""
    import numpy as np

    rf, _ = fit_surrogate(space, trials, "accuracy")
    assert rf is not None
    rows = np.array([c.get_array() for c in space.sample_configuration(20)])
    mean, spread = predict_with_spread(rf, rows)

    assert np.allclose(mean, rf.predict(rows))
    assert (spread >= 0).all()


def test_a_values_callable_that_raises_loses_only_that_trial(space, trials):
    """One unreadable trial must leave the fit intact."""
    def sometimes(t):
        if t.trial % 5 == 0:
            raise KeyError("no reading for this one")
        return t.scores["accuracy"]

    data = pair_trials_with_scores(space, trials, "accuracy", values=sometimes)

    assert len(data) == len([t for t in trials if t.trial % 5])


# The figure's "Predicted" means the search's own surrogate belief.

def test_a_search_that_models_nothing_has_no_surrogate_to_rebuild(optimizer, space, trials):
    """Random search fits nothing, and the caller learns that instead of getting
    a stand-in."""
    assert optimizer.refit_surrogate(space, trials, "accuracy") == (None, None)


def test_the_slice_still_draws_without_one(optimizer, space, trials):
    """`compute_incumbent_slice` draws without a surrogate."""
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "depth")
    positions, mu, sigma = sl.positions, sl.mu, sl.sigma

    assert len(positions) == len(mu) == len(sigma) > 1


@pytest.mark.parametrize("strategy,expected", [("rf", "RandomForest"),
                                               ("gp", "GaussianProcess")])
def test_a_run_is_read_through_its_own_model(space, trials, strategy, expected):
    """The surrogate's class and settings survive the run."""
    from core.optimizers.smac import SMACOptimizer

    surrogate, warning = SMACOptimizer(search_strategy=strategy).refit_surrogate(
        space, trials, "accuracy")

    assert warning is None
    assert surrogate is not None
    assert type(surrogate._model).__name__ == expected


def test_the_forest_is_not_told_its_targets_are_already_logged(space, trials):
    """SMAC's `get_model` hardcodes `log_y=True`, which breaks on negative
    costs."""
    from smac.model.random_forest import RandomForest

    from core.optimizers.smac import SMACOptimizer

    surrogate, _ = SMACOptimizer(search_strategy="rf").refit_surrogate(
        space, trials, "accuracy")

    assert surrogate is not None
    assert cast(RandomForest, surrogate._model)._log_y is False


def test_predictions_come_back_in_the_metrics_units(space, trials):
    """An accuracy slice returned as cost would be upside down."""
    from core.optimizers.smac import SMACOptimizer

    optimizer = SMACOptimizer(search_strategy="rf")
    sl = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "depth")
    mu, eta = sl.mu, sl.eta

    observed = [t.scores["accuracy"] for t in trials]
    assert min(observed) - 0.5 <= min(mu) <= max(mu) <= max(observed) + 0.5
    assert min(observed) <= eta <= max(observed)


def test_a_measured_output_that_is_not_a_metric_keeps_its_own_units(space, trials):
    """Duration skips every metric conversion."""
    from core.optimizers.smac import SMACOptimizer

    sl = analytics.compute_incumbent_slice(
        SMACOptimizer(search_strategy="rf"), space, trials, "accuracy", "depth",
        values=lambda t: t.duration)
    mu = sl.mu

    durations = [t.duration for t in trials]
    assert min(durations) - 1 <= min(mu) <= max(mu) <= max(durations) + 1


# The figure only marks the incumbent because the slice passes through no other
# trial.


def _best(trials, metric="accuracy"):
    return max((t for t in trials if metric in t.scores),
               key=lambda t: t.scores[metric])


def test_the_incumbent_is_the_one_measured_point_on_the_slice(optimizer, space, trials):
    sliced = analytics.compute_incumbent_slice(optimizer, space, trials, "accuracy", "lr")

    assert sliced.incumbent is not None
    position, value = sliced.incumbent
    assert 0.0 <= position <= 1.0
    assert value == pytest.approx(_best(trials).scores["accuracy"])


def test_the_incumbents_position_is_its_own_value_not_a_grid_point(optimizer, space, trials):
    """Snapping to a grid point would move the marker off the incumbent's
    configuration."""
    sliced = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", n_points=5)
    assert sliced.incumbent is not None
    position, _ = sliced.incumbent

    expected = float(space["lr"].to_vector(_best(trials).config["lr"]))
    assert position == pytest.approx(expected)


def test_there_is_no_incumbent_to_mark_without_the_metric(optimizer, space, trials):
    sliced = analytics.compute_incumbent_slice(optimizer, space, trials, "f1", "lr")

    assert sliced.incumbent is None


def test_under_values_the_incumbent_carries_what_was_modelled(optimizer, space, trials):
    """`eta` becomes the best observed duration while accuracy still picks the
    incumbent."""
    sliced = analytics.compute_incumbent_slice(
        optimizer, space, trials, "accuracy", "lr", values=lambda t: t.duration)

    assert sliced.incumbent is not None
    _, value = sliced.incumbent
    assert value == pytest.approx(_best(trials).duration)


def test_a_categoricals_incumbent_sits_at_its_choice_index(optimizer, space, trials):
    sliced = analytics.compute_incumbent_slice(optimizer, space, trials, "accuracy", "kern")
    assert sliced.incumbent is not None
    position, _ = sliced.incumbent

    choices = list(space["kern"].choices)
    assert position == pytest.approx(choices.index(_best(trials).config["kern"]))
    assert position in [pytest.approx(p) for p in sliced.positions]
