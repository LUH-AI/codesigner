"""`compute_incumbent_slice` — the surrogate along one hyperparameter.

The figure it backs lets a reader state a belief about where the optimum is by
dragging a curve, and a belief is a density on ConfigSpace's *normalized*
representation. So the thing most worth pinning here is not the predictions —
those are a random forest's and are allowed to move — but the **axis**: that
each grid value comes back with the normalized coordinate SMAC would evaluate a
prior at, on a log-scaled hyperparameter as much as a linear one.
"""

import math
import random

import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from core.optimizers.base import (
    TrialResult, _pair_trials_with_scores, _predict_with_spread, fit_surrogate)
from core.optimizers import RandomOptimizer


@pytest.fixture
def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([
        Float("lr", (1e-4, 1e-1), log=True),   # the units trap lives here
        Integer("depth", (1, 20)),
        Categorical("kern", ["rbf", "linear", "poly"]),
    ])
    return cs


@pytest.fixture
def trials(space):
    """Thirty trials of a function with a peak in `lr` near 1e-2."""
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


# ── the axis, which is the whole point ───────────────────────────────────────

def test_a_log_hyperparameters_positions_are_evenly_spaced(optimizer, space, trials):
    """`lr` spans 1e-4 to 1e-1 logarithmically. Its *values* are log-spaced and
    its *positions* are not: they are evenly spaced across [0, 1], because that
    is the representation a prior is evaluated in.

    Drawing a belief against the native values instead would put its mass
    somewhere else entirely — the midpoint of this axis is lr ≈ 3e-3, not 0.05.
    """
    values, positions, _, _, _, warning = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "lr", n_points=5)

    assert warning is None
    assert positions == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert values[0] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-1)
    # The geometric middle, which is what an evenly-spaced position means here.
    assert values[2] == pytest.approx(math.sqrt(1e-4 * 1e-1))


def test_an_integer_hyperparameters_positions_are_not_evenly_spaced(optimizer, space, trials):
    """`_hp_grid` collapses an integer grid to its distinct values, so a
    request for nine points over 1..20 comes back shorter and unevenly placed.
    Positions are therefore computed per value rather than assumed to be a
    `linspace` — this test is what fails if that assumption creeps back.
    """
    values, positions, mu, _, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "depth", n_points=9)

    assert values == sorted(set(values)), "distinct and ordered"
    assert len(positions) == len(values) == len(mu)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert max(gaps) - min(gaps) > 1e-6, "not evenly spaced, and must not be treated as if it were"


def test_a_categoricals_positions_are_its_choice_indices(optimizer, space, trials):
    """ConfigSpace encodes a categorical as its index, so that is what comes
    back. A categorical has no curve to drag — the figure draws bars — and this
    is the signal the caller reads to know that."""
    values, positions, _, _, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "kern")

    assert values == ["rbf", "linear", "poly"]
    assert positions == [0.0, 1.0, 2.0]


# ── the lists the browser indexes in lockstep ────────────────────────────────

def test_every_returned_list_has_the_same_length(optimizer, space, trials):
    for name in ("lr", "depth", "kern"):
        values, positions, mu, sigma, _, _ = optimizer.compute_incumbent_slice(
            space, trials, "accuracy", name, n_points=17)
        assert len(values) == len(positions) == len(mu) == len(sigma) > 1, name


def test_the_spread_is_never_negative(optimizer, space, trials):
    _, _, _, sigma, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "lr", n_points=31)

    assert all(s >= 0 for s in sigma)


# ── what it slices through ───────────────────────────────────────────────────

def test_the_slice_finds_the_peak_it_was_given(optimizer, space, trials):
    """A sanity check on the surrogate rather than on the plumbing: the trials
    carry a peak near lr = 1e-2, and the slice should show one there."""
    values, _, mu, _, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "lr", n_points=61)

    peak = values[mu.index(max(mu))]
    assert 1e-3 < peak < 1e-1


def test_a_second_slice_cuts_through_the_same_configuration(optimizer, space, trials):
    """The constraint panel models a different quantity — trial duration — but
    has to be read against the objective panel above it, so both must cut the
    same line. The incumbent is picked by the metric either way, never by
    whatever is being modelled."""
    a_values, a_positions, a_mu, _, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "lr", n_points=21)
    b_values, b_positions, b_mu, _, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "lr", n_points=21, values=lambda t: t.duration)

    assert a_values == b_values
    assert a_positions == b_positions
    assert a_mu != b_mu, "a different quantity, or the values= argument did nothing"


def test_duration_is_modellable_although_no_metric_reports_it(optimizer, space, trials):
    """`duration` lives on TrialResult as a property, not in `scores` — the
    reason `values=` is a callable rather than a second metric name."""
    _, _, mu, _, _, warning = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "depth", n_points=9, values=lambda t: t.duration)

    assert warning is None
    assert all(v > 0 for v in mu), "seconds, so positive"


# ── degrading rather than raising ────────────────────────────────────────────

def test_an_unknown_hyperparameter_is_a_warning(optimizer, space, trials):
    values, positions, _, _, _, warning = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "nonesuch")

    assert (values, positions) == ([], [])
    assert "nonesuch" in warning


def test_too_few_trials_is_a_warning(optimizer, space, trials):
    values, _, _, _, _, warning = optimizer.compute_incumbent_slice(
        space, trials[:1], "accuracy", "lr")

    assert values == []
    assert warning


def test_a_metric_no_trial_carries_is_a_warning(optimizer, space, trials):
    values, _, _, _, _, warning = optimizer.compute_incumbent_slice(
        space, trials, "f1", "lr")

    assert values == []
    assert warning


# ── the lifted estimator ─────────────────────────────────────────────────────

def test_the_mean_is_the_forests_own_prediction(space, trials):
    """`_predict_with_spread` computes the mean from the same per-tree stack the
    spread needs, so a band and the line through it can never come from two
    different passes. That only holds if the mean really is `rf.predict`."""
    import numpy as np

    rf, _ = fit_surrogate(space, trials, "accuracy")
    rows = np.array([c.get_array() for c in space.sample_configuration(20)])
    mean, spread = _predict_with_spread(rf, rows)

    assert np.allclose(mean, rf.predict(rows))
    assert (spread >= 0).all()


def test_a_values_callable_that_raises_loses_only_that_trial(space, trials):
    """A trial the caller cannot read should drop out the way a config the space
    rejects does — one bad trial must not blank out the fit."""
    def sometimes(t):
        if t.trial % 5 == 0:
            raise KeyError("no reading for this one")
        return t.scores["accuracy"]

    data = _pair_trials_with_scores(space, trials, "accuracy", values=sometimes)

    assert len(data) == len([t for t in trials if t.trial % 5])


# ── Whose surrogate ──────────────────────────────────────────────────────────
#
# The figure says "Predicted" and means the search's own belief. Fitting a
# generic forest instead makes every run look alike and denies that the choice
# of surrogate mattered at all.

def test_a_search_that_models_nothing_has_no_surrogate_to_rebuild(optimizer, space, trials):
    """Random search fits nothing, so there is nothing of its own to show and
    the caller is told so rather than handed a stand-in dressed up as one."""
    assert optimizer.refit_surrogate(space, trials, "accuracy") == (None, None)


def test_the_slice_still_draws_without_one(optimizer, space, trials):
    """... and losing the fidelity must not cost the figure."""
    _, positions, mu, sigma, _, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "depth")

    assert len(positions) == len(mu) == len(sigma) > 1


@pytest.mark.parametrize("strategy,expected", [("rf", "RandomForest"),
                                               ("gp", "GaussianProcess")])
def test_a_run_is_read_through_its_own_model(space, trials, strategy, expected):
    """A Gaussian-process run is read through a Gaussian process. The object
    does not survive the run, but its class and settings do."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    surrogate, warning = SMACOptimizer(search_strategy=strategy).refit_surrogate(
        space, trials, "accuracy")

    assert warning is None
    assert type(surrogate._model).__name__ == expected


def test_the_forest_is_not_told_its_targets_are_already_logged(space, trials):
    """SMAC's own `get_model` hardcodes `log_y=True`, and a forest believing
    that exponentiates what it is given. Costs are negative under a metric with
    no upper bound, so refitting on them needs it off."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    surrogate, _ = SMACOptimizer(search_strategy="rf").refit_surrogate(
        space, trials, "accuracy")

    assert surrogate._model._log_y is False


def test_predictions_come_back_in_the_metrics_units(space, trials):
    """The model learns cost; the panel is labelled with the metric. An
    accuracy slice that came back as cost would be upside down."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    optimizer = SMACOptimizer(search_strategy="rf")
    _, _, mu, _, eta, _ = optimizer.compute_incumbent_slice(
        space, trials, "accuracy", "depth")

    observed = [t.scores["accuracy"] for t in trials]
    assert min(observed) - 0.5 <= min(mu) <= max(mu) <= max(observed) + 0.5
    assert min(observed) <= eta <= max(observed)


def test_a_measured_output_that_is_not_a_metric_keeps_its_own_units(space, trials):
    """What the constraint panel needs: a surrogate over trial duration, which
    no metric reports and which must not be run through a metric's conversion."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    _, _, mu, _, _, _ = SMACOptimizer(search_strategy="rf").compute_incumbent_slice(
        space, trials, "accuracy", "depth", values=lambda t: t.duration)

    durations = [t.duration for t in trials]
    assert min(durations) - 1 <= min(mu) <= max(mu) <= max(durations) + 1
