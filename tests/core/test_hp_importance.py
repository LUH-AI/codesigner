"""Contract for BaseOptimizer.compute_hp_importance (core/optimizers/base.py).

Estimates per-hyperparameter importance from completed trials, with a fallback
ladder (HyperSHAP → RandomForest surrogate → uniform) returning
(importance_dict, warning_or_None).
"""

from core.models import RandomForestModel
from core.optimizers import RandomOptimizer, TrialResult


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
