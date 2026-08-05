"""Contract for the built-in models (core/models/random_forest.py, svm.py).

Pins each model's advertised search space and the fit_predict contract: one
prediction per validation row, of the dataset's own label type, deterministically
for a fixed seed. A model is never asked for a score and never sees the
validation labels — scoring is core.metrics' job.
"""

import pytest

from core.metrics import score_all
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
def test_fit_predict_returns_one_label_per_validation_row(model_cls, iris_splits):
    """One prediction per row, drawn from the labels the model was trained on."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(model_cls().get_config_space(seed=0).get_default_configuration())

    y_pred = model_cls().fit_predict(cfg, X_train, y_train, X_val, seed=0)

    assert len(y_pred) == len(X_val)
    assert set(y_pred) <= set(y_train)


@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_predictions_score_against_every_metric(model_cls, iris_splits, metrics):
    """What the model returns is scoreable by the application's own metrics."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(model_cls().get_config_space(seed=0).get_default_configuration())

    y_pred = model_cls().fit_predict(cfg, X_train, y_train, X_val, seed=0)
    scores = score_all(y_val, y_pred, metrics)

    assert set(scores) == set(metrics)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_fit_predict_is_deterministic(iris_splits):
    """The same config + seed reproduces identical predictions."""
    X_train, X_val, y_train, y_val = iris_splits
    cfg = dict(RandomForestModel().get_config_space(seed=0).get_default_configuration())

    a = RandomForestModel().fit_predict(cfg, X_train, y_train, X_val, seed=0)
    b = RandomForestModel().fit_predict(cfg, X_train, y_train, X_val, seed=0)

    assert list(a) == list(b)
