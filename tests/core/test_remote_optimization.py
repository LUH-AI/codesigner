"""`optimize()` only needs a name, a config space and `fit_predict`, so a remote
model needs no optimizer changes.
"""

import sys
import textwrap

import pytest

from core.metrics import metrics_for, resolve
from core.modelhost import model_session
from core.splits import holdout
from core.optimizers import GridOptimizer, RandomOptimizer
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS

from tests.conftest import launch_local

#: Regression metrics would fail on class labels.
METRICS = resolve(metrics_for("classification"))

MODEL = '''
# /// script
# dependencies = ["ConfigSpace", "scikit-learn"]
# ///
from ConfigSpace import ConfigurationSpace, Integer
from sklearn.neighbors import KNeighborsClassifier

from codesigner_model import BaseModel


class Neighbours(BaseModel):
    name = "Neighbours"

    def get_config_space(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("n_neighbors", (1, 3), default=1)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        clf = KNeighborsClassifier(n_neighbors=int(config["n_neighbors"]))
        clf.fit(X_train, y_train)
        return clf.predict(X_val)
'''


@pytest.fixture
def remote_model(tmp_path, iris_splits):
    """`remote_model` runs sklearn in another process."""
    path = tmp_path / "neighbours.py"
    path.write_text(textwrap.dedent(MODEL), encoding="utf-8")
    X_train, X_val, y_train, y_val = iris_splits
    with model_session(launch_local(sys.executable, path),
                       holdout(X_train, y_train, X_val, y_val), seed=0) as model:
        yield model


def test_random_search_optimizes_a_model_in_another_process(remote_model, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = RandomOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=3, seed=0)

    assert len(result.trials) == 3
    assert all(t.run_info["status"] == STATUS_SUCCESS for t in result.trials)
    assert result.best_score is not None and result.best_score > 0.5
    assert all(t.duration >= 0.0 for t in result.trials)


def test_the_optimizer_sees_the_remote_model_as_an_ordinary_one(remote_model, iris_splits):
    assert remote_model.name == "Neighbours"
    assert list(remote_model.get_config_space(seed=0).keys()) == ["n_neighbors"]
    X_train, X_val, y_train, y_val = iris_splits
    assert len(remote_model.fit_predict({"n_neighbors": 1}, X_train, y_train, X_val)) == len(X_val)


def test_grid_search_exhausts_the_remote_space(remote_model, iris_splits):
    """`GridOptimizer` introspects a space rebuilt from the child's description."""
    X_train, X_val, y_train, y_val = iris_splits

    result = GridOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0)

    # three n_neighbors values, three grid points
    assert len(result.trials) == 3
    assert {t.config["n_neighbors"] for t in result.trials} == {1, 2, 3}


def test_importance_is_computed_from_the_rebuilt_space(remote_model, iris_splits):
    """HyperSHAP gets a ConfigurationSpace rebuilt from JSON over a pipe."""
    X_train, X_val, y_train, y_val = iris_splits

    result = RandomOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=3, seed=0)

    assert set(result.hyperparameter_importance) == set(METRICS)
    assert "n_neighbors" in result.hyperparameter_importance["accuracy"]


def _flaky_model(tmp_path):
    """`_flaky_model` writes a model raising on `n_neighbors == 2`."""
    source = textwrap.dedent(MODEL).replace(
        "        clf = KNeighborsClassifier",
        '        if int(config["n_neighbors"]) == 2:\n'
        '            raise ValueError("two is unlucky")\n'
        "        clf = KNeighborsClassifier")
    path = tmp_path / "flaky.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_a_run_survives_a_model_that_fails_on_one_configuration(tmp_path, iris_splits):
    """`max_failures` defaults to 1, so the run sets 10 and the grid continues
    past the crashed trial."""
    path = _flaky_model(tmp_path)
    X_train, X_val, y_train, y_val = iris_splits

    with model_session(launch_local(sys.executable, path),
                       holdout(X_train, y_train, X_val, y_val), seed=0) as model:
        result = GridOptimizer().optimize(
            model, X_train, y_train, X_val, y_val,
            metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0,
            stopping={"max_trials": 50, "max_failures": 10})

    by_config = {t.config["n_neighbors"]: t for t in result.trials}
    assert len(by_config) == 3, "the search continued past the failure"
    assert by_config[2].run_info["status"] == STATUS_CRASHED
    assert by_config[2].scores["accuracy"] == 0.0
    assert "two is unlucky" in by_config[2].run_info["additional_info"]["error"]
    assert by_config[1].run_info["status"] == STATUS_SUCCESS


def test_by_default_the_first_failure_ends_the_run(tmp_path, iris_splits):
    """`DEFAULT_MAX_FAILURES` is 1, so the grid stops at the failed trial."""
    from core.optimizers.base import STOPPED_BY_ALL_FAILING

    path = _flaky_model(tmp_path)
    X_train, X_val, y_train, y_val = iris_splits

    with model_session(launch_local(sys.executable, path),
                       holdout(X_train, y_train, X_val, y_val), seed=0) as model:
        result = GridOptimizer().optimize(
            model, X_train, y_train, X_val, y_val,
            metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0)

    assert result.metadata["stopped_by"] == STOPPED_BY_ALL_FAILING
    # The grid stopped on 2 before running 3.
    assert {t.config["n_neighbors"] for t in result.trials} == {1, 2}
    failed = result.trials[-1]
    assert failed.failed is True
    assert "two is unlucky" in failed.failure


def test_the_models_own_traceback_crosses_the_process_boundary(tmp_path, iris_splits):
    """`evaluate_trial` stores the traceback the harness sent, because a
    parent-side `format_exc()` would show the client waiting."""
    path = _flaky_model(tmp_path)
    X_train, X_val, y_train, y_val = iris_splits

    with model_session(launch_local(sys.executable, path),
                       holdout(X_train, y_train, X_val, y_val), seed=0) as model:
        result = GridOptimizer().optimize(
            model, X_train, y_train, X_val, y_val,
            metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0,
            stopping={"max_trials": 50, "max_failures": 10})

    failed = next(t for t in result.trials if t.failed)
    assert "Traceback (most recent call last)" in failed.traceback
    # the frame from the model's own file
    assert "flaky.py" in failed.traceback
    assert 'raise ValueError("two is unlucky")' in failed.traceback
