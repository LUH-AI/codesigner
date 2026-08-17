"""Contract for BaseOptimizer.compute_hp_importance (core/optimizers/base.py).

Estimates per-hyperparameter importance from completed trials, with a fallback
ladder (HyperSHAP → RandomForest surrogate → uniform) returning
(importance_dict, warning_or_None).
"""

from unittest.mock import patch

from core.models import RandomForestModel
from core.optimizers import RandomOptimizer, TrialResult
from core.optimizers.base import BaseOptimizer, fit_surrogate


def _trials(iris_splits, metrics):
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


def test_insufficient_trials_returns_uniform_with_warning():
    """Fewer than two usable trials → uniform weights and a warning about trials."""
    cs = RandomForestModel().get_config_space(seed=0)
    params = list(cs.keys())
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    imp, warning = RandomOptimizer().compute_hp_importance(cs, one, "accuracy", seed=0)
    assert set(imp) == set(params)
    assert all(abs(v - 1.0 / len(params)) < 1e-9 for v in imp.values())
    assert warning and "trials" in warning.lower()


def test_sufficient_trials_produce_normalized_importance(iris_splits, metrics):
    """Enough trials → a normalized distribution over the config-space hyperparameters."""
    cs = RandomForestModel().get_config_space(seed=0)
    imp, _ = RandomOptimizer().compute_hp_importance(
        cs, _trials(iris_splits, metrics), "accuracy", seed=0)
    assert set(imp) == set(cs.keys())
    assert all(v >= 0.0 for v in imp.values())
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_interactions_insufficient_trials_returns_empty_not_uniform():
    """Fewer than two usable trials → empty, with a warning — the same
    "not enough trials" rung `_compute_hp_game` returns for importance itself,
    since there's no `iv` to re-extract pairwise terms from."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    interactions, warning = RandomOptimizer().compute_hp_interactions(cs, one, "accuracy", seed=0)
    assert interactions == {}
    assert warning and "trials" in warning.lower()


def test_interactions_form_a_symmetric_square_grid(iris_splits, metrics):
    """Enough trials → one entry per hyperparameter pair (plus the diagonal),
    symmetric, and the diagonal matches compute_hp_importance's own order-1
    values before normalization — same HyperSHAP call, just unnormalized and
    signed rather than abs-valued and scaled to sum to 1."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    params = list(cs.keys())
    interactions, warning = RandomOptimizer().compute_hp_interactions(cs, trials, "accuracy", seed=0)
    assert warning is None
    assert set(interactions) == set(params)
    for a in params:
        assert set(interactions[a]) == set(params)
        for b in params:
            assert interactions[a][b] == interactions[b][a]


def test_fit_surrogate_insufficient_trials_returns_none_with_warning():
    """Fewer than two usable trials → no surrogate, with a warning — the
    same condition every compute_hp_* fallback ladder treats as "nothing to
    fit from," not a fit that happens to be bad."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    rf, warning = fit_surrogate(cs, one, "accuracy", seed=0)
    assert rf is None
    assert warning and "trials" in warning.lower()


def test_fit_surrogate_returns_a_usable_regressor(iris_splits, metrics):
    """Enough trials → a fitted RandomForestRegressor: feature_importances_
    is the exact thing _compute_hp_game's own fallback rung reads, and
    predict() is what a future surrogate-consumer (Partial Dependencies,
    Phase 6) needs that HyperSHAP's own explainer doesn't expose directly."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    rf, warning = fit_surrogate(cs, trials, "accuracy", seed=0)
    assert warning is None
    assert len(rf.feature_importances_) == len(cs.keys())
    sample = trials[0].config
    from ConfigSpace import Configuration
    prediction = rf.predict([Configuration(cs, values=sample).get_array()])
    assert isinstance(prediction[0], float)


def test_compute_hp_games_builds_the_explainer_once_per_metric_not_per_game(iris_splits, metrics):
    """tunability/sensitivity/mistunability explain the same trial history
    for a given metric — fitting the underlying surrogate 3 times over (once
    per game) instead of once was pure waste. One accuracy explainer + one
    f1 explainer, not three of each."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()

    with patch.object(BaseOptimizer, "_build_explainer",
                     wraps=opt._build_explainer) as spy:
        opt.compute_hp_games(cs, trials, ["accuracy", "f1"], seed=0)

    called_metrics = [call.args[2] for call in spy.call_args_list]
    assert called_metrics == ["accuracy", "f1"]


def test_ablation_insufficient_trials_returns_empty_not_uniform():
    """Fewer than two usable trials → empty, with a warning — unlike the three
    global games, there is no uniform-weights rung: a signed "how much did
    each hyperparameter help or hurt" has no honest uniform answer."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    ablation, warning = RandomOptimizer().compute_hp_ablation(
        cs, one, "accuracy", dict(cs.get_default_configuration()), seed=0)
    assert ablation == {}
    assert warning and "trials" in warning.lower()


def test_ablation_explains_one_trial_against_the_default(iris_splits, metrics):
    """Enough trials → a signed value per hyperparameter for the trial handed
    in, not a normalized distribution (no fallback ladder to normalize)."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    ablation, warning = RandomOptimizer().compute_hp_ablation(
        cs, trials, "accuracy", trials[-1].config, seed=0)
    assert warning is None
    assert set(ablation) == set(cs.keys())
    assert all(isinstance(v, float) for v in ablation.values())


def test_partial_dependence_insufficient_trials_returns_empty():
    """Fewer than two usable trials → nothing to fit a surrogate from, so
    grid/ice_lines/pdp are all empty rather than some default grid with no
    predictions to show on it."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, one, "accuracy", "n_estimators", seed=0)
    assert grid == [] and ice_lines == [] and pdp == []
    assert warning and "trials" in warning.lower()


def test_partial_dependence_integer_hp_grid_is_all_integers(iris_splits, metrics):
    """n_estimators is a UniformIntegerHyperparameter — the grid must never
    suggest a fractional trial count, unlike a plain linspace would."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "n_estimators", seed=0)
    assert warning is None
    assert all(isinstance(v, int) for v in grid)
    assert grid == sorted(grid)


def test_partial_dependence_categorical_hp_grid_is_its_choices(iris_splits, metrics):
    """A categorical hyperparameter's grid is its full, exact set of
    choices — there is nothing to interpolate between categories."""
    from core.models import SVMModel
    X_train, X_val, y_train, y_val = iris_splits
    cs = SVMModel().get_config_space(seed=0)
    trials = RandomOptimizer().optimize(
        SVMModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "kernel", seed=0)
    assert warning is None
    assert grid == list(cs["kernel"].choices)


def test_partial_dependence_ice_lines_and_pdp_share_the_grids_shape(iris_splits, metrics):
    """One ICE row per trial, each the same length as the grid; the PDP
    curve is the grid-wise mean of those rows, not some other reduction."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0)
    assert warning is None
    assert len(ice_lines) == len(trials)
    assert all(len(row) == len(grid) for row in ice_lines)
    assert len(pdp) == len(grid)
    for i in range(len(grid)):
        column = [row[i] for row in ice_lines]
        assert pdp[i] == sum(column) / len(column)
