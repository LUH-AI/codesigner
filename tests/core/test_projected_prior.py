"""Priors stated in a view of the search space: a bump around a point, a ridge
along a line, a slab around a plane, in HP Space or a PCA/PLS projection.

The shape is one function of a whole configuration, handed to SMAC as one prior
over the whole space and multiplied with the per-hyperparameter beliefs. See
core/projected_prior.py for the math these tests hold it to.
"""

import math
from types import SimpleNamespace
from typing import Any, List, cast

import numpy as np
import pytest
from ConfigSpace import (Categorical, ConfigurationSpace, EqualsCondition, Float,
                         Integer)

from core import priors as priors_module
from core import projected_prior as pp
from core.models import RandomForestModel
from core.models.svm import SVMModel
from core.projection import fit


def _record(points, mean=None, spread=None, width=0.1, **view):
    view = view or {"view": "axes", "hyperparameters": ["x", "y", "z"][:len(points[0])]}
    return dict(view, points=points,
                params={"mean": mean or [], "spread": spread or [], "width": width})


def _space():
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("x", (0.0, 1.0)), Float("y", (0.0, 1.0)), Float("z", (0.0, 1.0))])
    return cs


# ── the shape ────────────────────────────────────────────────────────────────

def test_points_that_span_less_than_they_claim_are_refused():
    """Two points in one place, or a third on the line through the other two,
    describe no line or plane; nor do more points than the view has axes."""
    with pytest.raises(pp.PriorError):
        pp.frame([[0.2, 0.2], [0.2, 0.2]])
    with pytest.raises(pp.PriorError):
        pp.frame([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    with pytest.raises(pp.PriorError):
        pp.frame([[0.1], [0.9]])


def test_one_point_is_a_round_bump():
    """Peak 1 at the point, the same height at the same distance in any
    direction, and exp(−½) one width away."""
    record = _record([[0.5, 0.5]], width=0.1)
    Z = np.array([[0.5, 0.5], [0.6, 0.5], [0.5, 0.4], [0.5 + 0.1 / math.sqrt(2)] * 2])

    w = pp.density(Z, record)

    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(math.exp(-0.5))
    assert w[2] == pytest.approx(w[1]) and w[3] == pytest.approx(w[1])


def test_two_points_are_a_ridge_along_their_line():
    """With a wide spread along the line the weight barely changes walking
    along it, and falls with the width walking across it.

    Setup: a ridge from (0.2, 0.2) to (0.8, 0.8), spread 100 along, width 0.05.
    Expect: equal weight anywhere on the line; exp(−½) one width off it.
    """
    record = _record([[0.2, 0.2], [0.8, 0.8]], mean=[0.5], spread=[100.0], width=0.05)
    off = 0.05 / math.sqrt(2)
    Z = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8], [0.5 + off, 0.5 - off]])

    w = pp.density(Z, record)

    assert w[:3] == pytest.approx([1.0] * 3, abs=1e-4)
    assert w[3] == pytest.approx(math.exp(-0.5), rel=1e-4)


def test_position_along_the_line_is_in_the_points_own_units():
    """0 is the first point and 1 the second, so a mean of 1 puts the peak on
    the second point whatever the line's length."""
    record = _record([[0.0, 0.0], [0.4, 0.0]], mean=[1.0], spread=[0.25], width=0.1)
    w = pp.density(np.array([[0.4, 0.0], [0.0, 0.0]]), record)

    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(math.exp(-0.5 * (1 / 0.25) ** 2))


def test_three_points_are_a_slab_around_their_plane():
    """Within the plane (wide spreads) the weight stays up; off it, it falls
    with the width."""
    record = _record([[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]],
                     mean=[0.5, 0.5], spread=[100.0, 100.0], width=0.1)
    Z = np.array([[0.1, 0.9, 0.5], [0.9, 0.2, 0.5], [0.5, 0.5, 0.6]])

    w = pp.density(Z, record)

    assert w[:2] == pytest.approx([1.0, 1.0], abs=1e-4)
    assert w[2] == pytest.approx(math.exp(-0.5), rel=1e-4)


# ── in HP Space and in projections ──────────────────────────────────────────

def test_one_point_in_hp_space_is_the_normals_each_axis_already_takes():
    """In HP Space a bump around one point is exactly the product of a Normal on
    each axis — the prior any hyperparameter can already be given on its own —
    on the same vectorized axis and in the same unnormalised form.

    Setup: the forest's space; a point on max_features and min_samples_split.
    Expect: the projected prior's weight equals the product of the two Normals
    `core.priors` evaluates, for every sampled configuration.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    record = {"view": "axes", "hyperparameters": ["max_features", "min_samples_split"],
              "points": [[0.3, 0.6]], "params": {"width": 0.15}}
    X = np.array([c.get_array() for c in cs.sample_configuration(200)])
    order = list(cs.keys())

    w = pp.weight(record, cs, X)

    expected = np.ones(len(X))
    for name, mu in zip(record["hyperparameters"], record["points"][0]):
        column = X[:, order.index(name)]
        density = priors_module.density_from("normal", {"mu": mu, "sigma": 0.15}, list(column))
        assert density is not None
        expected = expected * np.array([y for _, y in density])
    assert w == pytest.approx(expected, rel=1e-9)


def _trials(cs, n=12, seed=1):
    cs.seed(seed)
    configs = [dict(c) for c in cs.sample_configuration(n)]
    rng = np.random.default_rng(seed)
    return [SimpleNamespace(config=c) for c in configs], list(rng.random(n))


@pytest.mark.parametrize("method", ["pca", "pls"])
def test_a_projection_reads_smacs_arrays_as_the_figure_reads_configurations(method):
    """The prior evaluates SMAC's vectorized rows; the figure transforms
    configuration dicts. Both go through the projection's one encoding, so a
    configuration lands in the same place either way — logged, one-hot and
    standardised alike.

    Setup: the SVM's space (log-scaled floats, categoricals) and a fitted
    projection, frozen into a record as the page would.
    Expect: the record's view coordinates of each configuration's array equal
    `Fitted.transform` of its dict.
    """
    cs = SVMModel().get_config_space(seed=0)
    trials, scores = _trials(cs)
    fitted, warning = fit(cs, trials, scores, method)
    assert fitted is not None, warning
    d = len(fitted.labels)
    linear = fitted.linear_map()
    record = {"view": method, "map": {"columns": linear["columns"],
                                      "matrix": linear["matrix"][:d], "offset": linear["offset"][:d]},
              "points": [[0.0] * d], "params": {"width": 1.0}}
    cs.seed(7)
    configs = cs.sample_configuration(50)

    from_arrays = pp.view_coordinates(record, cs, np.array([c.get_array() for c in configs]))
    from_dicts = np.array(fitted.transform([dict(c) for c in configs]))[:, :d]

    assert from_arrays == pytest.approx(from_dicts, abs=1e-9)


def test_each_categorical_choice_gets_its_own_weight_in_a_projection():
    """No special handling: two configurations differing only in a categorical
    land in different places in PCA, so the prior weights them differently."""
    cs = SVMModel().get_config_space(seed=0)
    trials, scores = _trials(cs)
    fitted, _ = fit(cs, trials, scores, "pca")
    assert fitted is not None
    linear = fitted.linear_map()
    base = dict(cs.get_default_configuration())
    rows = []
    for kernel in ("rbf", "linear"):
        from ConfigSpace import Configuration
        rows.append(Configuration(cs, values=dict(base, kernel=kernel)).get_array())
    point = fitted.transform([dict(base, kernel="rbf")])[0][:2]
    record = {"view": "pca", "map": {"columns": linear["columns"], "matrix": linear["matrix"][:2],
                                     "offset": linear["offset"][:2]},
              "points": [point], "params": {"width": 0.5}}

    w = pp.weight(record, cs, np.array(rows))

    assert w[0] == pytest.approx(1.0)
    assert w[1] < w[0]


def test_an_inactive_axis_neither_helps_nor_hurts():
    """A hyperparameter its parent switched off has no value to weigh. On an
    HP Space axis it takes the prior's own mode, so the configuration is
    weighed by the axes it does have."""
    cs = ConfigurationSpace(seed=0)
    switch = Categorical("switch", ["on", "off"])
    child = Float("child", (0.0, 1.0))
    other = Float("other", (0.0, 1.0))
    cs.add([switch, child, other, EqualsCondition(child, switch, "on")])
    record = {"view": "axes", "hyperparameters": ["child", "other"],
              "points": [[0.2, 0.7]], "params": {"width": 0.1}}
    order = list(cs.keys())
    row = np.full(len(order), np.nan)
    row[order.index("switch")] = 1  # "off"
    row[order.index("other")] = 0.7

    assert pp.weight(record, cs, row[None, :])[0] == pytest.approx(1.0)


# ── as SMAC takes it ─────────────────────────────────────────────────────────

def _forest_record(width=0.08):
    return {"view": "axes", "hyperparameters": ["max_features", "min_samples_split"],
            "points": [[0.25, 0.75]], "params": {"width": width}, "label": "test"}


def test_smac_takes_it_as_a_prior_over_the_whole_space():
    """Names every hyperparameter in column order, which SMAC requires, and
    answers one non-negative density per row, coarsened when a forest asks."""
    cs = RandomForestModel().get_config_space(seed=0)
    prior = pp.smac_prior(cs, _forest_record())
    X = np.array([c.get_array() for c in cs.sample_configuration(100)])

    prior.validate_against(cs)
    values = prior.pdf(X)
    coarse = prior.pdf(X, resolution=4)

    assert values.shape == (100, 1) and (values >= 0).all()
    assert len(np.unique(coarse)) <= 4


def test_drawing_from_it_lands_near_its_peak():
    """SMAC's maximizer draws part of its candidates from a prior, which is how
    a narrow belief gets reached at all; drawn by resampling, the draws gather
    around the point."""
    cs = RandomForestModel().get_config_space(seed=0)
    record = _forest_record(width=0.05)
    prior = pp.smac_prior(cs, record)
    order = list(cs.keys())
    cols = [order.index(h) for h in record["hyperparameters"]]

    drawn = prior.sample(200, np.random.RandomState(0))
    uniform = cs.sample_configuration(200)

    def spread(configs):
        X = np.array([c.get_array() for c in configs])[:, cols]
        return np.linalg.norm(X - np.array(record["points"][0]), axis=1).mean()

    assert len(drawn) == 200
    assert spread(drawn) < 0.5 * spread(uniform)


def test_beliefs_combine_by_product():
    """The per-hyperparameter prior and a projected one in one prior: both have
    to hold, so the density is their product."""
    from smac.acquisition.weight import TabulatedPrior

    cs = RandomForestModel().get_config_space(seed=0)
    grid = [i / 50 for i in range(51)]
    table = TabulatedPrior(cs, {"n_estimators": [[x, math.exp(-0.5 * ((x - 0.4) / 0.1) ** 2)]
                                                 for x in grid]})
    projected = pp.smac_prior(cs, _forest_record())
    combined = pp.combined_prior(cs, [table, projected])
    X = np.array([c.get_array() for c in cs.sample_configuration(50)])

    combined.validate_against(cs)
    assert combined.pdf(X) == pytest.approx(table.pdf(X) * projected.pdf(X))
    assert len(combined.sample(20, np.random.RandomState(1))) == 20


# ── the search, under it ─────────────────────────────────────────────────────

@pytest.mark.slow
def test_the_search_is_steered_by_a_projected_prior(iris_splits, metrics):
    """A run applies it — and says so — and what SMAC would ask next moves
    toward a narrow one.

    Setup: a four-trial SMAC search on iris; then asked what it would run next
    with and without a narrow bump on two hyperparameters.
    Expect: the run's prior report names the projected prior as applied, and
    the asked configurations sit closer to the bump with it than without.
    """
    from core.optimizers import SMACOptimizer

    X_train, X_val, y_train, y_val = iris_splits
    record = _forest_record(width=0.05)
    result = SMACOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=4, seed=0,
        projected_priors=[record])
    report = result.metadata.get("prior") or {}
    assert report.get("applied") and report.get("projected") == ["test"]

    cs = RandomForestModel().get_config_space(seed=0)
    order = list(cs.keys())
    cols = [order.index(h) for h in record["hyperparameters"]]

    def distance(**extra):
        asked = SMACOptimizer().slice_challengers(cs, result.trials, "accuracy", seed=0, **extra)
        assert asked is not None
        X = np.array([c.get_array() if hasattr(c, "get_array") else
                      __import__("ConfigSpace").Configuration(cs, values=c).get_array()
                      for c in cast(List[Any], asked[0])])[:, cols]
        return np.linalg.norm(X - np.array(record["points"][0]), axis=1).mean()

    assert distance(projected_priors=[record]) < distance()


# ── what its card draws ─────────────────────────────────────────────────────

@pytest.mark.parametrize("points, kind", [
    ([[0.5, 0.5]], "radial"),
    ([[0.2, 0.2], [0.8, 0.8]], "ridge"),
    ([[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]], "slab")])
def test_the_card_draws_the_density_the_search_weighs_by(points, kind):
    """The card's picture is `density` evaluated in the prior's own frame: one
    profile across a bump, a field along and across a ridge, a field over a
    slab's plane — peaking at 1 where the parameters put the peak."""
    k = len(points) - 1
    record = _record(points, mean=[0.5] * k, spread=[0.3] * k, width=0.1)

    shape = pp.profile(record)

    assert shape["kind"] == kind
    values = shape["y"] if kind == "radial" else [v for row in shape["z"] for v in row]
    assert max(values) == pytest.approx(1.0, abs=0.02)
    if kind != "radial":
        assert shape["mode"][0] == pytest.approx(0.5)
        assert shape["points"][:2] == [[0.0, 0.0], [1.0, 0.0]]
