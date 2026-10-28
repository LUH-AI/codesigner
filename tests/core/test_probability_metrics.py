"""`metrics_for` adds ROC AUC and log loss only on request, and `usable` drops
them without probabilities.
"""

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from core.metrics import METRICS, metrics_for, score_all, usable
from core.model_source import inspect_model_source
from core.analytics import EAGER_MAX_COALITIONS
from core.registry import MODELS


def test_probability_metrics_come_with_probabilities():
    plain = metrics_for("classification")
    asked = metrics_for("classification", probabilities=True)

    assert "balanced_accuracy" in plain
    assert "roc_auc" not in plain and "log_loss" not in plain
    assert asked == plain + ["roc_auc", "log_loss"]
    assert metrics_for("regression", probabilities=True) == ["rmse", "mae", "r2"]


def test_usable_drops_what_a_model_without_probabilities_cannot_score():
    """`usable` keeps the order and any unknown metric name."""
    names = ["accuracy", "roc_auc", "f1", "log_loss", "someones_own"]

    assert usable(names, probabilities=False) == ["accuracy", "f1", "someones_own"]
    assert usable(names, probabilities=True) == names


def test_roc_auc_is_the_usual_binary_auc():
    """Two-class ROC AUC equals scikit-learn's AUC for the second class."""
    y = np.array(["no", "yes", "yes", "no", "yes"])
    proba = np.array([[.9, .1], [.3, .7], [.6, .4], [.8, .2], [.1, .9]])

    ours = METRICS["roc_auc"].fn(y, proba, ["no", "yes"])
    assert ours == pytest.approx(roc_auc_score(y == "yes", proba[:, 1]))


def test_roc_auc_skips_a_class_a_fold_cannot_rank():
    """`roc_auc` leaves a class absent from the fold out of the average instead
    of failing the trial."""
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([[.8, .1, .1], [.2, .7, .1], [.6, .3, .1], [.3, .6, .1]])

    assert METRICS["roc_auc"].fn(y, proba, ["a", "b", "c"]) == 1.0


def test_log_loss_is_lower_is_better_and_a_failed_trial_scores_the_class_balance():
    """`null_score` gives a trial without predictions the log loss of predicting
    the class shares."""
    metric = METRICS["log_loss"]
    y = np.array(["a", "a", "a", "b"])
    proba = np.array([[.9, .1], [.8, .2], [.7, .3], [.4, .6]])

    assert metric.fn(y, proba, ["a", "b"]) == pytest.approx(log_loss(y, proba, labels=["a", "b"]))
    assert not metric.higher_is_better
    assert metric.null_score(y) == pytest.approx(log_loss(y, [[.75, .25]] * 4, labels=["a", "b"]))


def test_every_metric_scores_a_perfect_prediction_at_its_best():
    y = np.array(["a", "b", "c", "a"])
    proba = np.eye(3)[[0, 1, 2, 0]] * 0.999 + 0.001 / 3
    scores = score_all(y, list(y), {n: METRICS[n] for n in
                                     metrics_for("classification", probabilities=True)},
                       y_proba=proba, classes=["a", "b", "c"])

    assert scores["balanced_accuracy"] == scores["roc_auc"] == 1.0
    assert scores["log_loss"] < 0.01


@pytest.mark.parametrize("defines, expected", [(True, True), (False, False)])
def test_the_source_says_whether_a_model_gives_probabilities(defines, expected):
    """`inspect_model_source` finds `fit_predict_proba` without running the
    upload."""
    method = ("    def fit_predict_proba(self, config, X_train, y_train, X_val, seed=0):\n"
              "        return [], [], []\n") if defines else ""
    source = ('from codesigner_model import BaseModel\n\n'
              'class M(BaseModel):\n    name = "M"\n'
              '    def get_config_space(self, seed=0): return None\n'
              '    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []\n'
              + method)
    info, error = inspect_model_source(source.encode())

    assert error is None and info is not None and info.has_proba is expected


@pytest.mark.parametrize("name", sorted(MODELS))
def test_class_weights_are_a_classification_choice(name):
    model = MODELS[name].__class__()
    for task in model.tasks:
        model.task = task
        space = model.get_config_space()
        if "class_weight" in space:
            assert task == "classification"
            assert list(space["class_weight"].choices) == ["none", "balanced"]


@pytest.mark.parametrize("name", sorted(set(MODELS) - {"SVM"}))
def test_every_new_model_can_afford_its_analytics_with_probabilities(name):
    """The coalition count 2^hp × 3 games × 7 metrics must fit
    `EAGER_MAX_COALITIONS`, which the six-hyperparameter SVM exceeds."""
    model = MODELS[name].__class__()
    model.task = model.tasks[0]
    n_hp = len(list(model.get_config_space().values()))
    metrics = len(metrics_for(model.task, probabilities=True))

    assert 2 ** n_hp * 3 * metrics <= EAGER_MAX_COALITIONS


def test_a_class_the_training_rows_lacked_is_scored_not_fatal():
    """`log_loss` treats an unseen validation label as zero probability and
    returns a large finite loss."""
    y = np.array(["a", "b", "c"])
    proba = np.array([[.6, .4], [.3, .7], [.5, .5]])

    with_unseen = METRICS["log_loss"].fn(y, proba, ["a", "b"])
    assert np.isfinite(with_unseen) and with_unseen > METRICS["log_loss"].fn(y[:2], proba[:2], ["a", "b"])
