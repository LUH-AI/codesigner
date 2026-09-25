"""Rebuilding a prior's density from the numbers that define it.

`core.priors` is the *only* implementation of each density. The figure asks the
server for its curve through `ui.views._prior_density` and the search gets its
table through `SMACOptimizer._apply_priors`, and both come through here — so
what the reader watches and what the optimizer weights by cannot disagree,
rather than merely being expected not to.

It was briefly two: the browser evaluated the density in JavaScript as well, so
a drag could redraw at pointer speed. That stopped being needed when the
parameterised shapes stopped being draggable, and one implementation is worth
more than the latency it costs.
"""

import math

import pytest

from core.priors import density_from

GRID = [i / 256 for i in range(257)]


def _peak(values):
    return max(values, key=lambda point: point[1])[0]


def test_a_normal_peaks_where_its_mean_is():
    """Within a grid step, because the mean need not land on a grid point —
    0.6 is not representable in 256ths."""
    for mu in (0.6, 0.25, 0.9):
        peak = _peak(density_from("normal", {"mu": mu, "sigma": 0.15}, GRID))
        assert abs(peak - mu) <= 1 / 256


def test_a_normal_is_symmetric_about_its_mean():
    values = dict(density_from("normal", {"mu": 0.5, "sigma": 0.2}, GRID))

    for offset in (0.125, 0.25, 0.375):
        assert math.isclose(values[0.5 - offset], values[0.5 + offset], rel_tol=1e-9)


def test_a_beta_peaks_at_its_analytic_mode():
    """(α−1)/(α+β−2), which is the check that catches an exponent off by one —
    the most likely way to mistranscribe this from the JS."""
    for alpha, beta in ((2, 5), (5, 2), (3, 3)):
        mode = (alpha - 1) / (alpha + beta - 2)
        peak = _peak(density_from("beta", {"alpha": alpha, "beta": beta}, GRID))
        assert abs(peak - mode) <= 1 / 256


def test_a_beta_below_one_stays_finite_at_the_ends():
    """α or β under one puts an asymptote at an endpoint. Unclamped this is
    `inf`, and one infinite weight makes every other configuration's relative
    preference zero."""
    values = density_from("beta", {"alpha": 0.5, "beta": 0.5}, GRID)

    assert all(math.isfinite(v) for _, v in values)


def test_a_hand_placed_curve_does_not_overshoot_its_points():
    """The reason the interpolation is monotone rather than a plain cubic. An
    overshoot is a preference the reader never placed, and past zero it is a
    negative density."""
    points = [[0.0, 0.1], [0.4, 1.0], [1.0, 0.2]]
    values = [v for _, v in density_from("tabulated", {"points": points}, GRID)]

    assert min(values) >= 0.1 - 1e-9
    assert max(values) <= 1.0 + 1e-9


def test_a_hand_placed_curve_passes_through_its_points():
    points = [[0.0, 0.2], [0.5, 0.9], [1.0, 0.4]]
    values = dict(density_from("tabulated", {"points": points}, GRID))

    for x, y in points:
        assert math.isclose(values[x], y, rel_tol=1e-9)


def test_stating_nothing_rebuilds_nothing():
    """A uniform prior multiplies the acquisition by a constant, which cannot
    reorder it. Returning a flat table instead would have the optimizer carry a
    weight that does nothing, and report that a prior was applied."""
    assert density_from("uniform", {}, GRID) is None
    assert density_from("something else", {}, GRID) is None
    # Fewer than two points is not a curve.
    assert density_from("tabulated", {"points": [[0.5, 1.0]]}, GRID) is None


def test_a_density_is_never_negative():
    for kind, params in (("normal", {"mu": 0.5, "sigma": 0.01}),
                         ("beta", {"alpha": 0.2, "beta": 3}),
                         ("tabulated", {"points": [[0.0, 0.0], [1.0, 1.0]]})):
        assert all(v >= 0 for _, v in density_from(kind, params, GRID)), kind


# ── the canary that says when this module can shrink ─────────────────────────

def test_configspace_still_mangles_sigma_on_a_log_axis():
    """Pins the defect that keeps the Normal and Beta formulas above alive.

    `ConfigSpacePrior` is the class SMAC provides for stated priors and would
    replace them outright — `NormalFloatHyperparameter(log=True)` is a Gaussian
    on the log axis, which is exactly what the figure draws. But σ is passed
    through `UnitScaler.vectorize_size`, which applies the hyperparameter's log
    transform to it. σ is a scale, not a position, so that is a category error,
    and it shows: the effective width is |log10(σ)| / log-span, so σ = 1 has no
    width at all and σ and 1/σ are the same distribution.

    **When this test fails, ConfigSpace has fixed it.** At that point delete the
    `normal` and `beta` branches of `_values_for` and build a `ConfigSpacePrior`
    instead, keeping only the freeform interpolation. Until then the formulas
    here are the correct implementation, not a duplicate of a working one.
    """
    import numpy as np

    try:
        from ConfigSpace.hyperparameters.hp_components import UnitScaler
    except ImportError:  # pragma: no cover - layout differs across versions
        from ConfigSpace.hyperparameters._hp_components import UnitScaler

    scaler = UnitScaler(np.float64(1e-4), np.float64(1e0), log=True,
                        dtype=np.float64)

    # A unit standard deviation vanishes.
    assert float(scaler.vectorize_size(np.float64(1.0))) < 1e-4

    # And σ is indistinguishable from 1/σ, because only |log σ| survives.
    narrow = float(scaler.vectorize_size(np.float64(0.5)))
    wide = float(scaler.vectorize_size(np.float64(2.0)))
    assert abs(narrow - wide) < 1e-3, (narrow, wide)


def test_a_gaussian_on_a_log_axis_is_log_normal_in_native_units():
    """Why one formula serves both kinds of hyperparameter.

    The density is stated on the vectorized axis, and for a log hyperparameter
    that axis is log space — so the same expression that gives a Gaussian on a
    linear hyperparameter gives a log-normal on a log one. That is the right
    belief to hold about a learning rate, and it means nothing here needs to
    know whether a transform is in play.
    """
    from ConfigSpace import UniformFloatHyperparameter

    hp = UniformFloatHyperparameter("lr", 1e-4, 1e0, log=True)
    grid = [i / 512 for i in range(513)]
    values = density_from("normal", {"mu": 0.5, "sigma": 0.1}, grid)

    peak_native = hp.to_value(max(values, key=lambda point: point[1])[0])

    # The peak sits at the geometric centre of the range, not the arithmetic
    # one — which is what makes it a log-normal.
    assert peak_native == pytest.approx(math.sqrt(1e-4 * 1e0), rel=1e-2)
    assert peak_native < (1e-4 + 1e0) / 2
