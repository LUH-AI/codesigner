"""Classification and regression: what decides the task, and what follows from it."""

import copy

import numpy as np
import pytest

from core import io, tasks
from core.metrics import METRICS, metrics_for, to_cost
from core.model_source import inspect_model_source
from core.models import RandomForestModel, SVMModel
from core.registry import MODELS, canonical_model_name
from core.splits import cross_validation

from tests.conftest import DATASETS_DIR


# ── the target ───────────────────────────────────────────────────────────────

def test_regression_needs_a_numeric_target():
    """What: a text target cannot be regressed on; a numeric one can be either.
    How: asks `target_problem` of both."""
    assert tasks.target_problem(np.array(["a", "b"]), tasks.REGRESSION)
    assert tasks.target_problem(np.array(["a", "b"]), tasks.CLASSIFICATION) is None
    assert tasks.target_problem(np.array([1, 2]), tasks.REGRESSION) is None


def test_a_snapshot_without_a_task_is_classification():
    """What: files from before tasks existed read as classification.
    How: asks `task_of` of a snapshot with no task, and with an unknown one."""
    assert tasks.task_of({"evaluation": {}}) == tasks.CLASSIFICATION
    assert tasks.task_of({}) == tasks.CLASSIFICATION
    assert tasks.task_of({"evaluation": {"task": "regression"}}) == tasks.REGRESSION


# ── metrics ──────────────────────────────────────────────────────────────────

def test_each_task_has_its_own_metrics():
    """What: classification keeps its four metrics, in their order, then
    balanced accuracy; regression gets RMSE, MAE and R². How: compares
    `metrics_for` for each task."""
    assert metrics_for(tasks.CLASSIFICATION) == [
        "accuracy", "f1", "precision", "recall(macro)", "balanced_accuracy"]
    assert metrics_for(tasks.REGRESSION) == ["rmse", "mae", "r2"]


def test_regression_metrics_point_the_right_way():
    """What: lower RMSE and MAE are better, higher R² is, and each turns into a
    cost that a better model has less of. How: compares the costs of a perfect
    and an imperfect prediction."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    good, bad = y, y + 1.0
    for name in metrics_for(tasks.REGRESSION):
        metric = METRICS[name]
        assert to_cost(metric, metric.fn(y, good)) < to_cost(metric, metric.fn(y, bad)), name


def test_a_failed_regression_trial_scores_like_predicting_the_mean():
    """What: a trial that produced nothing scores the mean predictor's error,
    not zero — zero RMSE would be a perfect score. How: compares each null
    score with the score of predicting the mean."""
    y = np.array([1.0, 2.0, 4.0, 9.0])
    mean = np.full_like(y, y.mean())
    for name in metrics_for(tasks.REGRESSION):
        metric = METRICS[name]
        assert metric.null_score(y) == pytest.approx(metric.fn(y, mean)), name


# ── splitting ────────────────────────────────────────────────────────────────

def test_a_regression_split_is_never_stratified():
    """What: integer-valued regression targets are not treated as classes.
    How: a target where every value occurs exactly twice stratifies as
    classification — each fold gets one of each — and not as regression."""
    X = np.arange(40).reshape(-1, 1)
    y = np.repeat(np.arange(20), 2)

    def per_fold(task):
        splits = cross_validation(X, y, 2, seed=0, task=task)
        return [sorted(y[val]) for _, val in splits.folds]

    assert per_fold(tasks.CLASSIFICATION) == [list(range(20))] * 2
    assert per_fold(tasks.REGRESSION) != [list(range(20))] * 2


# ── models ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls", [RandomForestModel, SVMModel])
def test_the_built_ins_regress(model_cls):
    """What: told the task is regression, a built-in predicts numbers.
    How: fits the diabetes data with the default configuration."""
    X, y = io._load_frame(DATASETS_DIR / "diabetes.csv")
    model = model_cls()
    model.task = tasks.REGRESSION
    config = dict(model.get_config_space().get_default_configuration())

    predictions = np.asarray(model.fit_predict(config, X[:300], y[:300], X[300:]))

    assert predictions.shape == (len(X) - 300,)
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))


def test_the_svm_tunes_epsilon_only_when_regressing():
    """What: the tube width is a regression hyperparameter.
    How: compares the SVM's search space under each task."""
    model = SVMModel()
    assert "epsilon" not in model.get_config_space()
    model.task = tasks.REGRESSION
    assert "epsilon" in model.get_config_space()


def test_building_an_experiment_tells_a_copy_its_task(models, metrics, optimizers):
    """What: the registry's shared instance is not mutated; the experiment's
    copy is told the task. How: builds a regression snapshot read-only."""
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
    """What: files naming "SVM Classifier" open on the renamed model.
    How: canonicalises the old name and resolves it through the registry."""
    assert canonical_model_name("SVM Classifier") == "SVM"
    assert canonical_model_name("Something Else") == "Something Else"


def test_an_uploaded_model_says_what_it_supports():
    """What: `tasks` is read from the source without running it, and a model
    that says nothing classifies. How: inspects three small sources."""
    base = ("from codesigner_model import BaseModel\n\n"
            "class M(BaseModel):\n    name = 'M'\n{line}"
            "    def get_config_space(self, seed=0): return None\n"
            "    def fit_predict(self, config, X_train, y_train, X_val, seed=0): return []\n")

    said_nothing, _ = inspect_model_source(base.format(line="").encode())
    both, _ = inspect_model_source(
        base.format(line="    tasks = ('classification', 'regression')\n").encode())
    _, error = inspect_model_source(base.format(line="    tasks = ('ranking',)\n").encode())

    assert said_nothing.tasks == ("classification",)
    assert both.tasks == ("classification", "regression")
    assert "ranking" in error
