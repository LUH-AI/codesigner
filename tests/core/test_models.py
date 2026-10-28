"""Built-in models predict one label per validation row, deterministically per
seed, and `core.metrics` does the scoring.
"""

from typing import cast

import pytest
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace
from ConfigSpace.hyperparameters import NumericalHyperparameter

from core.metrics import score_all
from core.models import RandomForestModel, SVMModel


def _bounds(cs: ConfigurationSpace, name: str) -> tuple:
    hp = cast(NumericalHyperparameter, cs[name])
    return hp.lower, hp.upper


def test_random_forest_config_space():
    """`RandomForestModel` declares four documented bounded hyperparameters."""
    cs = RandomForestModel().get_config_space(seed=0)
    assert set(cs.keys()) == {"n_estimators", "max_depth", "min_samples_split", "max_features"}
    assert _bounds(cs, "n_estimators") == (10, 500)
    assert cs["n_estimators"].default_value == 100
    assert _bounds(cs, "max_depth") == (2, 50)
    assert _bounds(cs, "min_samples_split") == (0.01, 0.5)
    assert _bounds(cs, "max_features") == (0.1, 1.0)


def test_svm_config_space():
    """`SVMModel` declares six hyperparameters and documented kernels."""
    cs = SVMModel().get_config_space(seed=0)
    assert set(cs.keys()) == {"C", "kernel", "gamma", "degree", "tol", "max_iter"}
    assert set(cast(CategoricalHyperparameter, cs["kernel"]).choices) == {"rbf", "linear", "poly", "sigmoid"}
    assert set(cast(CategoricalHyperparameter, cs["gamma"]).choices) == {"scale", "auto"}


@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_fit_predict_returns_one_label_per_validation_row(model_cls, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(model_cls().get_config_space(seed=0).get_default_configuration())

    y_pred = model_cls().fit_predict(cfg, X_train, y_train, X_val, seed=0)

    assert len(y_pred) == len(X_val)
    assert set(y_pred) <= set(y_train)


@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_predictions_score_against_every_metric(model_cls, iris_splits, metrics):
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(model_cls().get_config_space(seed=0).get_default_configuration())

    y_pred = model_cls().fit_predict(cfg, X_train, y_train, X_val, seed=0)
    scores = score_all(y_val, y_pred, metrics)

    assert set(scores) == set(metrics)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_fit_predict_is_deterministic(iris_splits):
    """Seeded fits repeat."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(RandomForestModel().get_config_space(seed=0).get_default_configuration())

    a = RandomForestModel().fit_predict(cfg, X_train, y_train, X_val, seed=0)
    b = RandomForestModel().fit_predict(cfg, X_train, y_train, X_val, seed=0)

    assert list(a) == list(b)
