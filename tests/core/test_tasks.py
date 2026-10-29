"""`test_tasks` covers `core.tasks` and what the chosen task implies."""

import copy

import numpy as np
import pytest

from core import datasets, io, tasks
from core.metrics import METRICS, metrics_for, to_cost
from core.model_source import inspect_model_source
from core.models import RandomForestModel, SVMModel
from core.registry import MODELS, canonical_model_name
from core.splits import cross_validation

from tests.conftest import DATASETS_DIR


def test_regression_needs_a_numeric_target():
    """`target_problem` only returns a reason for a text target under
    regression."""
    assert tasks.target_problem(np.array(["a", "b"]), tasks.REGRESSION)
    assert tasks.target_problem(np.array(["a", "b"]), tasks.CLASSIFICATION) is None
    assert tasks.target_problem(np.array([1, 2]), tasks.REGRESSION) is None


def test_a_snapshot_without_a_task_is_classification():
    """`task_of` reads a missing task as classification."""
    assert tasks.task_of({"evaluation": {}}) == tasks.CLASSIFICATION
    assert tasks.task_of({}) == tasks.CLASSIFICATION
    assert tasks.task_of({"evaluation": {"task": "regression"}}) == tasks.REGRESSION


def test_each_task_has_its_own_metrics():
    """`metrics_for` lists balanced accuracy after the four classification
    metrics."""
    assert metrics_for(tasks.CLASSIFICATION) == [
        "accuracy", "f1", "precision", "recall(macro)", "balanced_accuracy"]
    assert metrics_for(tasks.REGRESSION) == ["rmse", "mae", "r2"]


def test_regression_metrics_point_the_right_way():
    """`to_cost` falls for every regression metric as the prediction improves."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    good, bad = y, y + 1.0
    for name in metrics_for(tasks.REGRESSION):
        metric = METRICS[name]
        assert to_cost(metric, metric.fn(y, good)) < to_cost(metric, metric.fn(y, bad)), name


def test_a_failed_regression_trial_scores_like_predicting_the_mean():
    """`null_score` is the mean predictor's error, because zero RMSE is perfect."""
    y = np.array([1.0, 2.0, 4.0, 9.0])
    mean = np.full_like(y, y.mean())
    for name in metrics_for(tasks.REGRESSION):
        metric = METRICS[name]
        assert metric.null_score(y) == pytest.approx(metric.fn(y, mean)), name


def test_a_regression_split_is_never_stratified():
    """`cross_validation` stratifies integer targets only for classification."""
    X = np.arange(40).reshape(-1, 1)
    y = np.repeat(np.arange(20), 2)

    def per_fold(task):
        splits = cross_validation(X, y, 2, seed=0, task=task)
        return [sorted(y[val]) for _, val in splits.folds]

    assert per_fold(tasks.CLASSIFICATION) == [list(range(20))] * 2
    assert per_fold(tasks.REGRESSION) != [list(range(20))] * 2


@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_the_built_ins_regress(model_cls):
    """`fit_predict` returns floats for every built-in on a regression task."""
    X, y = datasets.load_frame(DATASETS_DIR / "diabetes.csv")
    model = model_cls()
    model.task = tasks.REGRESSION
    config = dict(model.get_config_space().get_default_configuration())

    predictions = np.asarray(model.fit_predict(config, X[:300], y[:300], X[300:]))

    assert predictions.shape == (len(X) - 300,)
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))


def test_the_svm_tunes_epsilon_only_when_regressing():
    """Epsilon, the tube width, is a regression-only hyperparameter."""
    model = SVMModel()
    assert "epsilon" not in model.get_config_space()
    model.task = tasks.REGRESSION
    assert "epsilon" in model.get_config_space()


def test_building_an_experiment_tells_a_copy_its_task(models, metrics, optimizers):
    """`build_experiment` sets the task on a copy, leaving the registry's model
    untouched."""
    snapshot = io.normalize({
        "version": "0.1.0", "name": "r", "model_name": "Random Forest", "model_path": "",
        "optimizer_name": "Random Search", "optimizer_params": {},
        "metric_names": ["rmse"], "seed": 0, "dataset_path": "", "result": None})
    snapshot["evaluation"]["task"] = tasks.REGRESSION

    _, exp = io.build_experiment(copy.deepcopy(snapshot), METRICS, MODELS, optimizers,
                                 read_only=True)

    assert exp["task"] == tasks.REGRESSION
    assert exp["model"].task == tasks.REGRESSION
    assert exp["model"] is not MODELS["Random Forest"]
    assert MODELS["Random Forest"].task == tasks.CLASSIFICATION


def test_the_old_svm_name_still_resolves():
    assert canonical_model_name("SVM Classifier") == "SVM"
    assert canonical_model_name("Something Else") == "Something Else"


def test_an_uploaded_model_says_what_it_supports():
    """`inspect_model_source` reads `tasks` statically, defaults to
    classification and rejects unknown tasks."""
    base = ("from codesigner_model import BaseModel\n\n"
            "class M(BaseModel):\n    name = 'M'\n{line}"
            "    def get_config_space(self, seed=0): return None\n"
            "    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []\n")

    said_nothing, _ = inspect_model_source(base.format(line="").encode())
    both, _ = inspect_model_source(
        base.format(line="    tasks = ('classification', 'regression')\n").encode())
    _, error = inspect_model_source(base.format(line="    tasks = ('ranking',)\n").encode())

    assert said_nothing is not None and both is not None
    assert said_nothing.tasks == ("classification",)
    assert both.tasks == ("classification", "regression")
    assert error is not None
    assert "ranking" in error
