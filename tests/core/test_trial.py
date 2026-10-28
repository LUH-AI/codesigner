"""`evaluate_trial` serves every optimizer and turns a raising model into a
failed trial.
"""

import numpy as np
import pytest

from core.metrics import metrics_for, resolve
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS
from core.optimizers.trial import evaluate_trial
from core.splits import cross_validation, holdout

#: Classification metrics, since every split here has class labels.
METRICS = resolve(metrics_for("classification"))

X_TRAIN = np.array([[0.0], [1.0]])
Y_TRAIN = np.array(["a", "b"], dtype=object)
X_VAL = np.array([[2.0], [3.0]])
Y_VAL = np.array(["a", "b"], dtype=object)
SPLITS = holdout(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)


class Perfect:

    name = "Perfect"
    last_cpu_time: float

    def get_config_space(self, seed=0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        return ["a", "b"]


class Broken:
    name = "Broken"

    def get_config_space(self, seed=0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        raise RuntimeError("this configuration explodes")


class WrongLength:
    name = "Wrong length"

    def get_config_space(self, seed=0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        return ["a"]          # too short


def _run(model):
    return evaluate_trial(model, {"a": 1}, SPLITS, METRICS, seed=0)


def test_a_good_trial_is_scored_and_marked_successful():
    scores, run_info = _run(Perfect())

    assert scores["accuracy"] == 1.0
    assert run_info["status"] == STATUS_SUCCESS
    assert run_info["additional_info"] == {}
    assert run_info["time"] >= 0.0


def test_a_model_that_raises_becomes_a_recorded_failure():
    """The optimizer keeps searching past an unusable configuration."""
    scores, run_info = _run(Broken())

    assert scores == {name: 0.0 for name in METRICS}
    assert run_info["status"] == STATUS_CRASHED
    assert "this configuration explodes" in run_info["additional_info"]["error"]


def test_unscoreable_predictions_are_a_failure_not_a_crash():
    scores, run_info = _run(WrongLength())

    assert scores == {name: 0.0 for name in METRICS}
    assert run_info["status"] == STATUS_CRASHED
    assert "could not be scored" in run_info["additional_info"]["error"]


def test_every_requested_metric_is_present_even_on_failure():
    """Optimizers then read scores by primary metric."""
    scores, _ = _run(Broken())
    assert set(scores) == set(METRICS)


def test_a_model_that_timed_itself_overrides_the_cpu_reading():
    """`evaluate_trial` trusts `last_cpu_time`, since another process may have
    done the work."""
    model = Perfect()
    model.last_cpu_time = 42.0

    _, run_info = _run(model)

    assert run_info["cpu_time"] == 42.0
