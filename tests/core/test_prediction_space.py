"""`sample_prediction_space` predicts one even sample of the search space per
metric and places it in every cube view, with an outline per projection.
"""

from typing import cast, List

import numpy as np
import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from ConfigSpace.hyperparameters import NumericalHyperparameter

from core import analytics
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer, TrialResult
from core.projection import fit


def _trials(iris_splits, metrics):
    """`_trials` runs twelve seeded random-search trials of a forest on iris."""
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=12, seed=0).trials


def _space():
    """`_space` mixes a log float, an integer and a categorical."""
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("C", (0.01, 100.0), log=True), Integer("n", (1, 10)),
            Categorical("kernel", ["a", "b", "c"])])
    return cs


def _synthetic_trials(cs, count=20):
    """`_synthetic_trials` scores each trial by a smooth function of `C`."""
    rng = np.random.default_rng(0)
    trials = []
    for i, config in enumerate(cs.sample_configuration(count)):
        score = float(1.0 - abs(np.log10(config["C"])) / 4 + rng.normal(0, 0.01))
        trials.append(TrialResult(trial=i + 1, config=dict(config), scores={"accuracy": score},
                                  score=score, incumbent_score=score, incumbent_config={}))
    return trials


@pytest.mark.parametrize("method", ["pca", "pls"])
def test_a_configuration_projects_where_a_trial_running_it_would(iris_splits, metrics, method):
    """`Fitted.transform` maps the trials' configurations onto their coordinates
    to 1e-9.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    fitted, warning = fit(cs, trials, [t.scores["accuracy"] for t in trials], method)

    assert warning is None
    assert fitted is not None
    again = fitted.transform([t.config for t in trials])
    assert np.allclose(again, fitted.coordinates, atol=1e-9)


@pytest.mark.parametrize("method", ["pca", "pls"])
def test_every_legal_configuration_lands_inside_the_outline(iris_splits, metrics, method):
    from scipy.spatial import ConvexHull

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    answer = analytics.sample_prediction_space(
        RandomOptimizer(), cs, trials, "accuracy", seed=0, n=1024)
    projection = answer["projections"][method]
    points = np.array(projection["coordinates"])

    low, high = projection["outlines"]["1"]
    assert points[:, 0].min() >= low - 1e-6 and points[:, 0].max() <= high + 1e-6

    for k, outline in ((2, np.array(projection["outlines"]["2"])),
                       (3, None)):
        if k == 3:
            o = projection["outlines"]["3"]
            outline = np.column_stack([o["x"], o["y"], o["z"]])
        assert outline is not None
        hull = ConvexHull(outline)
        offsets = points[:, :k] @ hull.equations[:, :k].T + hull.equations[:, k]
        assert offsets.max() <= 1e-6, f"{method} {k}D"


def test_the_outline_is_the_box_and_no_larger(iris_splits, metrics):
    """A box's image under a linear map is spanned by its corners, so the
    outline is exact.
    """
    import itertools

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    fitted, _w = fit(cs, trials, [t.scores["accuracy"] for t in trials], "pca")
    assert fitted is not None
    numeric = cast(List[NumericalHyperparameter], list(cs.values()))
    corners = [dict(zip([hp.name for hp in cs.values()], ends))
               for ends in itertools.product(*[(hp.lower, hp.upper) for hp in numeric])]

    positions = [row[0] for row in fitted.transform(corners)]
    low, high = cast(List[float], fitted.outline(1))
    assert min(positions) == pytest.approx(low, abs=1e-9)
    assert max(positions) == pytest.approx(high, abs=1e-9)


def test_the_sample_covers_the_space_evenly_on_its_own_scales():
    """A Sobol sample is far more even than the two-point tolerance, so a
    failure points to a wrong scale.
    """
    cs = _space()
    answer = analytics.sample_prediction_space(
        RandomOptimizer(), cs, _synthetic_trials(cs), "accuracy", seed=0, n=4096)
    names = answer["hp_names"]
    columns = {name: [row[names.index(name)] for row in answer["values"]] for name in names}

    assert len(answer["values"]) == len(answer["mu"]) == len(answer["sigma"]) == 4096
    assert sum(c < 1 for c in columns["C"]) / 4096 == pytest.approx(0.5, abs=0.02)
    for kernel in "abc":
        assert columns["kernel"].count(kernel) / 4096 == pytest.approx(1 / 3, abs=0.02)
    assert all(isinstance(n, int) and 1 <= n <= 10 for n in columns["n"])


def test_the_sample_follows_the_experiments_seed():
    cs = _space()
    trials = _synthetic_trials(cs)
    opt = RandomOptimizer()

    first = analytics.sample_prediction_space(opt, cs, trials, "accuracy", seed=0, n=256)
    again = analytics.sample_prediction_space(opt, cs, trials, "accuracy", seed=0, n=256)
    other = analytics.sample_prediction_space(opt, cs, trials, "accuracy", seed=1, n=256)

    assert first["values"] == again["values"]
    assert first["values"] != other["values"]


def test_what_expected_improvement_needs_comes_with_it():
    """`sample_prediction_space` returns everything the page's expected
    improvement needs."""
    cs = _space()
    trials = _synthetic_trials(cs)
    answer = analytics.sample_prediction_space(
        RandomOptimizer(), cs, trials, "accuracy", seed=0, n=256)

    assert answer["eta"] == pytest.approx(max(t.scores["accuracy"] for t in trials), abs=1e-12)
    assert answer["higher_is_better"] is True
    assert all(s is not None and s >= 0 for s in answer["sigma"])
    assert answer["smooth"] is False


@pytest.mark.parametrize("method", ["pca", "pls"])
def test_the_linear_map_is_the_projection(iris_splits, metrics, method):
    """The page places proposed configurations with `linear_map` alone, so the
    map must match `transform` to 1e-9.
    """
    import math

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    fitted, _w = fit(cs, trials, [t.scores["accuracy"] for t in trials], method)
    assert fitted is not None
    linear = fitted.linear_map()

    for trial, expected in zip(trials, fitted.coordinates):
        row = []
        for column in linear["columns"]:
            value = trial.config.get(column["name"])
            if column["value"] is not None:
                raw = 1.0 if str(value) == column["value"] else 0.0
            else:
                assert value is not None
                raw = math.log10(value) if column["log"] else float(value)
            row.append((raw - column["mean"]) / column["spread"] if column["spread"] else 0.0)
        placed = np.array(linear["matrix"]) @ np.array(row) + np.array(linear["offset"])
        assert np.allclose(placed, expected, atol=1e-9)


def test_each_sample_carries_its_vectorized_position():
    """`vectors` contains each sample's ConfigSpace array, the axis priors are
    stated on."""
    from ConfigSpace import Configuration

    cs = _space()
    answer = analytics.sample_prediction_space(
        RandomOptimizer(), cs, _synthetic_trials(cs), "accuracy", seed=0, n=64)
    names = answer["hp_names"]
    for values, vector in zip(answer["values"], answer["vectors"]):
        expected = Configuration(cs, values=dict(zip(names, values))).get_array()
        assert np.allclose(vector, expected, atol=1e-4)
