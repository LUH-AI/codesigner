"""`core.analytics` falls back from HyperSHAP to a RandomForest surrogate to
uniform weights.
"""

from typing import cast
from unittest.mock import patch

import pytest
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from core import analytics
from core.analytics import games
from core.analytics import fit_surrogate
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer, TrialResult


def _trials(iris_splits, metrics):
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


def test_insufficient_trials_returns_uniform_with_warning():
    cs = RandomForestModel().get_config_space(seed=0)
    params = list(cs.keys())
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    imp, warning, *_ = analytics.compute_hp_game(cs, one, "accuracy", "tunability", seed=0)
    assert set(imp) == set(params)
    assert all(abs(v - 1.0 / len(params)) < 1e-9 for v in imp.values())
    assert warning and "trials" in warning.lower()


def test_sufficient_trials_produce_normalized_importance(iris_splits, metrics):
    """Importance normalizes over the config space."""
    cs = RandomForestModel().get_config_space(seed=0)
    imp, *_ = analytics.compute_hp_game(
        cs, _trials(iris_splits, metrics), "accuracy", "tunability", seed=0)
    assert set(imp) == set(cs.keys())
    assert all(v >= 0.0 for v in imp.values())
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_a_game_that_finds_nothing_reports_nothing_rather_than_zeros():
    """A saturating objective makes HyperSHAP return all zeros, and the empty
    result with a warning keeps the pie from drawing nothing. Uniform weights
    mean undecided, and an empty result means nothing found.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    flat = [TrialResult(trial=i, config=dict(cs.sample_configuration()),
                        scores={"accuracy": 0.7}, score=0.7,
                        incumbent_score=0.7, incumbent_config={})
            for i in range(1, 7)]

    imp, warning, *_ = analytics.compute_hp_game(cs, flat, "accuracy", "tunability", seed=0)

    assert imp == {}
    assert warning and "no signal" in warning
    uniform = 1.0 / len(list(cs.keys()))
    assert uniform not in imp.values(), "must not be confused with the uniform rung"


def test_a_game_that_finds_nothing_reports_no_interactions_either():
    """All-zero contributions contain no pairwise structure, and the heatmap shows
    its caption."""
    cs = RandomForestModel().get_config_space(seed=0)
    flat = [TrialResult(trial=i, config=dict(cs.sample_configuration()),
                        scores={"accuracy": 0.7}, score=0.7,
                        incumbent_score=0.7, incumbent_config={})
            for i in range(1, 7)]

    _imp, warning, interactions, *_ = analytics.compute_hp_game(
        cs, flat, "accuracy", "tunability", seed=0)

    assert interactions == {}
    assert warning and "no signal" in warning


def test_interactions_insufficient_trials_returns_empty_not_uniform():
    """Without an `iv`, no pairwise terms exist to extract."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    _imp, warning, interactions, *_ = analytics.compute_hp_game(
        cs, one, "accuracy", "tunability", seed=0)
    assert interactions == {}
    assert warning and "trials" in warning.lower()


def test_interactions_form_a_symmetric_square_grid(iris_splits, metrics):
    """The diagonal contains the signed, unnormalized order-1 values from the same
    HyperSHAP call."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    params = list(cs.keys())
    _imp, warning, interactions, *_ = analytics.compute_hp_game(
        cs, trials, "accuracy", "tunability", seed=0)
    assert warning is None
    assert set(interactions) == set(params)
    for a in params:
        assert set(interactions[a]) == set(params)
        for b in params:
            assert interactions[a][b] == interactions[b][a]


def test_fit_surrogate_insufficient_trials_returns_none_with_warning():
    """Too few trials means nothing to fit, as in every fallback ladder."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    rf, warning = fit_surrogate(cs, one, "accuracy", seed=0)
    assert rf is None
    assert warning and "trials" in warning.lower()


def test_fit_surrogate_returns_a_usable_regressor(iris_splits, metrics):
    """The fallback reads `feature_importances_`, and partial dependence calls
    `predict()`."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    rf, warning = fit_surrogate(cs, trials, "accuracy", seed=0)
    assert warning is None
    assert rf is not None
    assert len(rf.feature_importances_) == len(cs.keys())
    sample = trials[0].config
    from ConfigSpace import Configuration
    prediction = rf.predict([Configuration(cs, values=sample).get_array()])
    assert isinstance(prediction[0], float)


def test_compute_hp_games_builds_the_explainer_once_per_metric_not_per_game(iris_splits, metrics):
    """The three games share one explainer per metric."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    with patch.object(games, "build_explainer", wraps=games.build_explainer) as spy:
        analytics.compute_hp_games(cs, trials, ["accuracy", "f1"], seed=0)

    called_metrics = [call.args[2] for call in spy.call_args_list]
    assert called_metrics == ["accuracy", "f1"]


def test_the_fallback_surrogate_is_also_fitted_once_per_metric_not_per_game(iris_splits, metrics):
    """The RandomForest fallback is also shared per metric. Breaking HyperSHAP's
    three game methods sends six game calls to the fallback, which must fit
    twice.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    def explode(self, *a, **k):
        raise RuntimeError("no shapiq today")

    with patch("hypershap.HyperSHAP.tunability", explode), \
         patch("hypershap.HyperSHAP.sensitivity", explode), \
         patch("hypershap.HyperSHAP.mistunability", explode), \
         patch("core.analytics.games.fit_surrogate",
               wraps=fit_surrogate) as fits:
        games = analytics.compute_hp_games(cs, trials, ["accuracy", "f1"], seed=0)

    fitted_metrics = [call.args[2] for call in fits.call_args_list]
    assert fitted_metrics == ["accuracy", "f1"], "one fallback fit per metric"
    # The fallback still yields real answers.
    for game in analytics.HP_GAMES:
        importance, warning, interactions, moebius = (games[game][0]["accuracy"],
                                                      games[game][1]["accuracy"],
                                                      games[game][2]["accuracy"],
                                                      games[game][3]["accuracy"])
        assert set(importance) == set(cs.keys())
        assert warning and "falling back" in warning
        assert interactions == {}, "a plain feature_importances_ has no pairwise structure"
        assert moebius == [], "nor any coalition structure"


# Each game yields a Möbius decomposition once, and the importance figures
# convert it to FSII.


def test_the_moebius_transform_is_truncation_independent(iris_splits, metrics):
    """FSII order-1 terms shift when the order rises, while Möbius values stay
    fixed and can be stored once. Pinning both halves keeps the contrast that
    justifies the design.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = games.build_explainer(cs, trials, "accuracy", 0)
    n = len(list(cs.keys()))

    from shapiq import ExactComputer
    import hypershap.games as hs_games
    import hypershap.task as hs_task
    from hypershap.utils import Aggregation, RandomConfigSpaceSearcher

    # The facade's default is 10,000 draws.
    source = explainer.hs.explanation_task
    task = hs_task.TunabilityExplanationTask(
        config_space=cs, surrogate_model=source.surrogate_model,
        baseline_config=cs.get_default_configuration())
    game = hs_games.TunabilityGame(
        explanation_task=task,
        cs_searcher=RandomConfigSpaceSearcher(
            explanation_task=task, n_samples=10_000,
            mode=Aggregation.MAX, seed=0))
    ec = ExactComputer(n_players=game.get_num_hyperparameters(), game=game)

    shallow = ec(index="Moebius", order=2).get_n_order(order=1).dict_values
    deep = ec(index="Moebius", order=n).get_n_order(order=1).dict_values
    assert shallow == deep, "Möbius must not move with the truncation order"

    fsii_2 = ec(index="FSII", order=2).get_n_order(order=1).dict_values
    fsii_n = ec(index="FSII", order=n).get_n_order(order=1).dict_values
    assert fsii_2 != fsii_n, "FSII does move — which is why it is not used here"


def test_moebius_covers_every_coalition_of_every_size(iris_splits, metrics):
    """The decomposition drops the empty coalition, a constant offset."""
    cs = RandomForestModel().get_config_space(seed=0)
    params = set(cs.keys())
    trials = _trials(iris_splits, metrics)

    _imp, warning, _inter, moebius, _total = analytics.compute_hp_game(
        cs, trials, "accuracy", "tunability", seed=0)

    assert warning is None
    assert len(moebius) == 2 ** len(params) - 1
    assert {len(row["members"]) for row in moebius} == set(range(1, len(params) + 1))
    for row in moebius:
        assert set(row["members"]) <= params
        assert isinstance(row["value"], float)
    assert len({tuple(sorted(row["members"])) for row in moebius}) == len(moebius), "no duplicates"


def test_moebius_terms_are_ranked_by_magnitude(iris_splits, metrics):
    """Views take a top-k slice, making order part of the contract."""
    cs = RandomForestModel().get_config_space(seed=0)
    _i, _w, _x, moebius, _total = analytics.compute_hp_game(
        cs, _trials(iris_splits, metrics), "accuracy", "tunability", seed=0)

    magnitudes = [abs(row["value"]) for row in moebius]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_ablation_insufficient_trials_returns_empty_not_uniform():
    """A signed help-or-hurt value has no honest uniform answer."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    ablation, warning = analytics.compute_hp_ablation(
        cs, one, "accuracy", dict(cs.get_default_configuration()), seed=0)
    assert ablation == {}
    assert warning and "trials" in warning.lower()


def test_ablation_explains_one_trial_against_the_default(iris_splits, metrics):
    """Ablation returns signed values without normalizing."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    ablation, warning = analytics.compute_hp_ablation(
        cs, trials, "accuracy", trials[-1].config, seed=0)
    assert warning is None
    assert set(ablation) == set(cs.keys())
    assert all(isinstance(v, float) for v in ablation.values())


def test_the_explanation_surrogate_actually_uses_the_seed_it_is_given(iris_splits, metrics):
    """Two seeds must give two different fits for ablation and the global games.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    a, warn_a = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=0)
    b, warn_b = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=7)
    assert warn_a is None and warn_b is None
    assert a != b, "ablation ignored its seed"

    imp_a, *_ = analytics.compute_hp_game(cs, trials, "accuracy", "tunability", seed=0)
    imp_b, *_ = analytics.compute_hp_game(cs, trials, "accuracy", "tunability", seed=7)
    assert imp_a != imp_b, "the global games ignored their seed"


def test_seed_zero_keeps_hypershaps_own_default_estimator(iris_splits, metrics):
    """The explicit estimator equals hypershap's own default at seed 0, and
    stored results stay valid.
    """
    from sklearn.ensemble import RandomForestRegressor
    from hypershap import ExplanationTask, HyperSHAP
    from core.analytics import pair_trials_with_scores

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    data = pair_trials_with_scores(cs, trials, "accuracy")
    params = list(cs.keys())

    def order1(task):
        iv = HyperSHAP(task).tunability()
        return {params[i]: v for (i,), v in iv.get_n_order(order=1).dict_values.items()}

    theirs = order1(ExplanationTask.from_data(cs, data))
    ours = order1(ExplanationTask.from_data(
        cs, data, base_model=RandomForestRegressor(n_estimators=100, random_state=0)))
    assert theirs == ours


def test_partial_dependence_insufficient_trials_returns_empty():
    """Too few trials leaves grid, ICE lines and PDP empty."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, one, "accuracy", "n_estimators", seed=0)
    assert grid == [] and ice_lines == [] and pdp == []
    assert warning and "trials" in warning.lower()


def test_partial_dependence_integer_hp_grid_is_all_integers(iris_splits, metrics):
    """An integer grid never suggests a fractional count."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "n_estimators", seed=0)
    assert warning is None
    assert all(isinstance(v, int) for v in grid)
    assert grid == sorted(grid)


def test_partial_dependence_categorical_hp_grid_is_its_choices(iris_splits, metrics):
    """Categories have nothing to interpolate between."""
    from core.models import SVMModel
    X_train, X_val, y_train, y_val = iris_splits
    cs = SVMModel().get_config_space(seed=0)
    trials = RandomOptimizer().optimize(
        SVMModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "kernel", seed=0)
    assert warning is None
    assert grid == list(cast(CategoricalHyperparameter, cs["kernel"]).choices)


def test_partial_dependence_log_hp_grid_is_spaced_on_the_log_scale():
    """A native linspace on `C` would spend 19 of 20 points in the sparse upper
    tail. The test checks a constant ratio between consecutive points.
    """
    from core.models import SVMModel
    from core.analytics.surrogate import hp_grid

    grid = hp_grid(SVMModel().get_config_space(seed=0)["C"], 20)

    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(100.0)
    assert grid[1] == pytest.approx(0.016238, abs=1e-6), "a native linspace would put 5.27 here"
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_partial_dependence_linear_hp_grid_is_unchanged():
    """Linear grids must still match `linspace(lower, upper)`, on which many
    stored expectations rest.
    """
    import numpy as np

    from core.analytics.surrogate import hp_grid

    cs = RandomForestModel().get_config_space(seed=0)
    for name in cs.keys():
        hp = cast(NumericalHyperparameter, cs[name])
        native = np.linspace(hp.lower, hp.upper, 20)
        expected = (sorted({int(round(v)) for v in native})
                    if type(hp).__name__ == "UniformIntegerHyperparameter"
                    else [float(v) for v in native])
        assert hp_grid(hp, 20) == pytest.approx(expected), name


def test_partial_dependence_ice_lines_and_pdp_share_the_grids_shape(iris_splits, metrics):
    """The PDP is the grid-wise mean of the ICE rows."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0)
    assert warning is None
    assert len(ice_lines) == len(trials)
    assert all(len(row) == len(grid) for row in ice_lines)
    assert len(pdp) == len(grid)
    for i in range(len(grid)):
        column = [cast(float, row[i]) for row in ice_lines]
        assert pdp[i] == sum(column) / len(column)


def test_partial_dependence_cap_limits_the_curves_drawn(iris_splits, metrics):
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    _, uncapped, _, _ = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0, max_ice_curves=0)
    _, capped, _, _ = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0, max_ice_curves=3)

    assert len(uncapped) == len(trials)
    assert len(capped) == 3


def test_partial_dependence_cap_bounds_the_work_not_just_the_picture():
    """With a cap of 2 the PDP equals the mean of the first and last ICE rows,
    proving unsampled trials go unpredicted.
    """
    from ConfigSpace import ConfigurationSpace, Float

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0.0, 1.0), default=0.5), Float("b", (0.0, 1.0), default=0.5)])
    trials = [TrialResult(trial=i, config={"a": i / 10, "b": 1 - i / 10},
                          scores={"accuracy": i / 10}, score=i / 10,
                          incumbent_score=i / 10, incumbent_config={})
              for i in range(1, 11)]

    grid, ice, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=2)

    assert warning is None
    assert len(ice) == 2, "first and last only"
    for i in range(len(grid)):
        assert pdp[i] == pytest.approx((cast(float, ice[0][i]) + cast(float, ice[1][i])) / 2)


def test_partial_dependence_cap_above_the_trial_count_changes_nothing():
    from ConfigSpace import ConfigurationSpace, Float

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0.0, 1.0), default=0.5)])
    trials = [TrialResult(trial=i, config={"a": i / 10}, scores={"accuracy": i / 10},
                          score=i / 10, incumbent_score=i / 10, incumbent_config={})
              for i in range(1, 6)]

    _, _, uncapped, _ = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=0)
    _, _, generous, _ = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=500)

    assert uncapped == generous


def test_partial_dependence_batching_matches_predicting_one_at_a_time(iris_splits, metrics):
    """Batching removed sklearn's per-call overhead and must leave every number
    unchanged.
    """
    from ConfigSpace import Configuration

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0)
    assert warning is None

    rf, _ = fit_surrogate(cs, trials, "accuracy", 0)
    assert rf is not None
    expected = []
    for t in trials:
        row = []
        for value in grid:
            values = dict(t.config)
            values["max_depth"] = value
            cfg = Configuration(cs, values=values)
            row.append(float(rf.predict([cfg.get_array()])[0]))
        expected.append(row)

    assert ice_lines == expected


def test_partial_dependence_holes_where_the_config_space_refuses_a_grid_value():
    """The test builds its own forbidden clause because no registry model has
    one.
    """
    from ConfigSpace import (Categorical, ConfigurationSpace,
                             ForbiddenAndConjunction, ForbiddenEqualsClause, Integer)

    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("a", (1, 3), default=1), Categorical("b", ["x", "y"], default="x")])
    cs.add(ForbiddenAndConjunction(ForbiddenEqualsClause(cs["a"], 2),
                                   ForbiddenEqualsClause(cs["b"], "y")))

    trials = [
        TrialResult(trial=1, config={"a": 1, "b": "x"}, scores={"accuracy": 0.5},
                    score=0.5, incumbent_score=0.5, incumbent_config={}),
        TrialResult(trial=2, config={"a": 3, "b": "y"}, scores={"accuracy": 0.7},
                    score=0.7, incumbent_score=0.7, incumbent_config={}),
        TrialResult(trial=3, config={"a": 1, "b": "y"}, scores={"accuracy": 0.6},
                    score=0.6, incumbent_score=0.6, incumbent_config={}),
    ]

    grid, ice_lines, pdp, warning = analytics.compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0)

    assert warning is None
    assert grid == [1, 2, 3]
    # b="y" only forbids a=2.
    assert all(v is not None for v in ice_lines[0])
    for row in ice_lines[1:]:
        assert row[grid.index(2)] is None
        assert row[grid.index(1)] is not None and row[grid.index(3)] is not None
    # The mean skips missing rows.
    assert pdp[grid.index(2)] == ice_lines[0][grid.index(2)]


def test_the_explanation_games_are_drawn_with_the_experiments_own_seed(iris_splits, metrics):
    """The searcher draws must follow the experiment's seed like the split,
    model, search and surrogate. The spy checks the seed directly because a
    small space finds the same maximum for any seed.
    """
    import importlib

    facade = importlib.import_module("hypershap.hypershap")
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    seen = []
    real = facade.RandomConfigSpaceSearcher

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    with patch.object(facade, "RandomConfigSpaceSearcher", spy):
        analytics.compute_hp_games(cs, trials, metrics, seed=7)

    assert seen, "the searcher is built at all"
    assert set(seen) == {7}, "and every game gets the experiment's seed"
