"""Contract for the built-in models (core/models/random_forest.py, svm.py).

Pins each model's advertised search space and the train_evaluate contract:
score every requested metric, deterministically for a fixed seed.
"""

import pytest

from core.models import RandomForestModel, SVMModel


def test_random_forest_config_space():
    """Random Forest advertises its four documented hyperparameters and bounds."""
    cs = RandomForestModel().get_config_space(seed=0)
    assert set(cs.keys()) == {"n_estimators", "max_depth", "min_samples_split", "max_features"}
    assert (cs["n_estimators"].lower, cs["n_estimators"].upper) == (10, 500)
    assert cs["n_estimators"].default_value == 100
    assert (cs["max_depth"].lower, cs["max_depth"].upper) == (2, 50)
    assert (cs["min_samples_split"].lower, cs["min_samples_split"].upper) == (0.01, 0.5)
    assert (cs["max_features"].lower, cs["max_features"].upper) == (0.1, 1.0)


def test_svm_config_space():
    """SVM advertises its six documented hyperparameters with the right kernels."""
    cs = SVMModel().get_config_space(seed=0)
    assert set(cs.keys()) == {"C", "kernel", "gamma", "degree", "tol", "max_iter"}
    assert set(cs["kernel"].choices) == {"rbf", "linear", "poly", "sigmoid"}
    assert set(cs["gamma"].choices) == {"scale", "auto"}


@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_train_evaluate_scores_every_metric(model_cls, iris_splits, metrics):
    """train_evaluate returns a numeric score in [0,1] for every requested metric."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(model_cls().get_config_space(seed=0).get_default_configuration())
    scores = model_cls().train_evaluate(cfg, X_train, y_train, X_val, y_val, metrics, seed=0)
    assert set(scores) == set(metrics)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_train_evaluate_is_deterministic(iris_splits, metrics):
    """The same config + seed reproduces identical scores."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(RandomForestModel().get_config_space(seed=0).get_default_configuration())
    a = RandomForestModel().train_evaluate(cfg, X_train, y_train, X_val, y_val, metrics, seed=0)
    b = RandomForestModel().train_evaluate(cfg, X_train, y_train, X_val, y_val, metrics, seed=0)
    assert a == b
