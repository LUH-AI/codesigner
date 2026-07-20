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
