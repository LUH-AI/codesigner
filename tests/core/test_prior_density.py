"""`core.priors` is the one density implementation, shared by the figure's
`prior_density` and `SMACOptimizer._apply_priors`.
"""

import math

import pytest

from core.priors import density_from

GRID = [i / 256 for i in range(257)]


def _peak(values):
    return max(values, key=lambda point: point[1])[0]


def test_a_normal_peaks_where_its_mean_is():
    for mu in (0.6, 0.25, 0.9):
        peak = _peak(density_from("normal", {"mu": mu, "sigma": 0.15}, GRID))
        assert abs(peak - mu) <= 1 / 256


def test_a_normal_is_symmetric_about_its_mean():
    density = density_from("normal", {"mu": 0.5, "sigma": 0.2}, GRID)
    assert density is not None
    values = {x: v for x, v in density}

    for offset in (0.125, 0.25, 0.375):
        assert math.isclose(values[0.5 - offset], values[0.5 + offset], rel_tol=1e-9)


def test_a_beta_peaks_at_its_analytic_mode():
    """The analytic mode catches an exponent off by one."""
    for alpha, beta in ((2, 5), (5, 2), (3, 3)):
        mode = (alpha - 1) / (alpha + beta - 2)
        peak = _peak(density_from("beta", {"alpha": alpha, "beta": beta}, GRID))
        assert abs(peak - mode) <= 1 / 256


def test_a_beta_below_one_stays_finite_at_the_ends():
    """Shapes below one diverge at the ends, and one infinite weight would zero
    every other preference."""
    values = density_from("beta", {"alpha": 0.5, "beta": 0.5}, GRID)

    assert values is not None
    assert all(math.isfinite(v) for _, v in values)


def test_a_hand_placed_curve_does_not_overshoot_its_points():
    """A plain cubic would overshoot, placing preferences nobody stated and,
    below zero, a negative density."""
    points = [[0.0, 0.1], [0.4, 1.0], [1.0, 0.2]]
    density = density_from("tabulated", {"points": points}, GRID)
    assert density is not None
    values = [v for _, v in density]

    assert min(values) >= 0.1 - 1e-9
    assert max(values) <= 1.0 + 1e-9


def test_a_hand_placed_curve_passes_through_its_points():
    points = [[0.0, 0.2], [0.5, 0.9], [1.0, 0.4]]
    density = density_from("tabulated", {"points": points}, GRID)
    assert density is not None
    values = {x: v for x, v in density}

    for x, y in points:
        assert math.isclose(values[x], y, rel_tol=1e-9)


def test_stating_nothing_rebuilds_nothing():
    """A uniform prior scales the acquisition by a constant, so `density_from`
    returns None instead of a weight that changes nothing."""
    assert density_from("uniform", {}, GRID) is None
    assert density_from("something else", {}, GRID) is None
    assert density_from("tabulated", {"points": [[0.5, 1.0]]}, GRID) is None


def test_a_density_is_never_negative():
    for kind, params in (("normal", {"mu": 0.5, "sigma": 0.01}),
                         ("beta", {"alpha": 0.2, "beta": 3}),
                         ("tabulated", {"points": [[0.0, 0.0], [1.0, 1.0]]})):
        values = density_from(kind, params, GRID)
        assert values is not None, kind
        assert all(v >= 0 for _, v in values), kind


def test_configspace_still_mangles_sigma_on_a_log_axis():
    """`UnitScaler.vectorize_size` log-transforms σ, so σ = 1 has no width and σ
    matches 1/σ. A ConfigSpace fix fails the test and lets `ConfigSpacePrior`
    replace the `normal` and `beta` branches of `_values_for`.
    """
    import numpy as np

    from ConfigSpace.hyperparameters.hp_components import UnitScaler

    scaler = UnitScaler(np.float64(1e-4), np.float64(1e0), log=True,
                        dtype=np.float64)

    # Unit σ vanishes.
    assert float(scaler.vectorize_size(np.float64(1.0))) < 1e-4

    # Only |log σ| survives.
    narrow = float(scaler.vectorize_size(np.float64(0.5)))
    wide = float(scaler.vectorize_size(np.float64(2.0)))
    assert abs(narrow - wide) < 1e-3, (narrow, wide)


def test_a_gaussian_on_a_log_axis_is_log_normal_in_native_units():
    """`density_from` works on the vectorized axis, and that axis is log space
    for a log hyperparameter.
    """
    from ConfigSpace import UniformFloatHyperparameter

    hp = UniformFloatHyperparameter("lr", 1e-4, 1e0, log=True)
    grid = [i / 512 for i in range(513)]
    values = density_from("normal", {"mu": 0.5, "sigma": 0.1}, grid)
    assert values is not None

    peak_native = hp.to_value(max(values, key=lambda point: point[1])[0])

    # A log-normal peaks at the geometric centre of the range.
    assert peak_native == pytest.approx(math.sqrt(1e-4 * 1e0), rel=1e-2)
    assert peak_native < (1e-4 + 1e0) / 2
