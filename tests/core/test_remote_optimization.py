"""A real optimizer, searching a model that lives in another process.

The point of the seam: `optimize()` reads a name, asks for a config space and
calls one method per trial. It cannot tell whether the training happened in this
interpreter or another one, so none of the three optimizers needed changing to
support this.
"""

import sys
import textwrap

import pytest

from core.metrics import METRICS
from core.modelhost import model_session
from core.optimizers import GridOptimizer, RandomOptimizer
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS

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
    """A real sklearn model, in its own process, over iris."""
    path = tmp_path / "neighbours.py"
    path.write_text(textwrap.dedent(MODEL), encoding="utf-8")
    X_train, X_val, y_train, y_val = iris_splits
    with model_session(sys.executable, path, X_train, y_train, X_val, seed=0) as model:
        yield model


def test_random_search_optimizes_a_model_in_another_process(remote_model, iris_splits):
    X_train, X_val, y_train, y_val = iris_splits

    result = RandomOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=3, seed=0)

    assert len(result.trials) == 3
    assert all(t.run_info["status"] == STATUS_SUCCESS for t in result.trials)
    # iris is easy; a real classifier should be well above chance
    assert result.best_score > 0.5
    assert all(t.duration >= 0.0 for t in result.trials)


def test_the_optimizer_sees_the_remote_model_as_an_ordinary_one(remote_model, iris_splits):
    """Same three members, so nothing in the loop is conditional on where the
    model runs."""
    assert remote_model.name == "Neighbours"
    assert list(remote_model.get_config_space(seed=0).keys()) == ["n_neighbors"]
    X_train, X_val, y_train, y_val = iris_splits
    assert len(remote_model.fit_predict({"n_neighbors": 1}, X_train, y_train, X_val)) == len(X_val)


def test_grid_search_exhausts_the_remote_space(remote_model, iris_splits):
    """Grid introspects the config space by hyperparameter type, and the space it
    introspects was rebuilt from the child's description."""
    X_train, X_val, y_train, y_val = iris_splits

    result = GridOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0)

    # three values for n_neighbors, so the grid is three points and no more
    assert len(result.trials) == 3
    assert {t.config["n_neighbors"] for t in result.trials} == {1, 2, 3}


def test_importance_is_computed_from_the_rebuilt_space(remote_model, iris_splits):
    """HyperSHAP needs a live ConfigurationSpace; the one it gets came over a
    pipe as JSON."""
    X_train, X_val, y_train, y_val = iris_splits

    result = RandomOptimizer().optimize(
        remote_model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric="accuracy", n_trials=3, seed=0)

    assert set(result.hyperparameter_importance) == set(METRICS)
    assert "n_neighbors" in result.hyperparameter_importance["accuracy"]


def test_a_run_survives_a_model_that_fails_on_one_configuration(tmp_path, iris_splits):
    """One bad configuration is a data point, not the end of the search."""
    source = textwrap.dedent(MODEL).replace(
        "        clf = KNeighborsClassifier",
        '        if int(config["n_neighbors"]) == 2:\n'
        '            raise ValueError("two is unlucky")\n'
        "        clf = KNeighborsClassifier")
    path = tmp_path / "flaky.py"
    path.write_text(source, encoding="utf-8")
    X_train, X_val, y_train, y_val = iris_splits

    with model_session(sys.executable, path, X_train, y_train, X_val, seed=0) as model:
        result = GridOptimizer().optimize(
            model, X_train, y_train, X_val, y_val,
            metrics=METRICS, primary_metric="accuracy", n_trials=50, seed=0)

    by_config = {t.config["n_neighbors"]: t for t in result.trials}
    assert len(by_config) == 3, "the search continued past the failure"
    assert by_config[2].run_info["status"] == STATUS_CRASHED
    assert by_config[2].scores["accuracy"] == 0.0
    assert "two is unlucky" in by_config[2].run_info["additional_info"]["error"]
    assert by_config[1].run_info["status"] == STATUS_SUCCESS
