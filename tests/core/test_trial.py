"""One trial: predict, score, and record what happened.

All three optimizers share this, so a run's history cannot depend on which one
produced it. Its job beyond scoring is to decide what a *failed* trial is: a
model that raises on one configuration is a fact about that configuration, not
the end of the run.
"""

import numpy as np
import pytest

from core.metrics import METRICS
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS
from core.optimizers.trial import evaluate_trial

X_TRAIN = np.array([[0.0], [1.0]])
Y_TRAIN = np.array(["a", "b"], dtype=object)
X_VAL = np.array([[2.0], [3.0]])
Y_VAL = np.array(["a", "b"], dtype=object)


class Perfect:
    """Predicts the validation labels it was never shown, by luck."""

    name = "Perfect"

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
        return ["a"]          # one prediction for two rows


def _run(model):
    return evaluate_trial(model, {"a": 1}, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, METRICS, seed=0)


def test_a_good_trial_is_scored_and_marked_successful():
    scores, run_info = _run(Perfect())

    assert scores["accuracy"] == 1.0
    assert run_info["status"] == STATUS_SUCCESS
    assert run_info["additional_info"] == {}
    assert run_info["time"] >= 0.0


def test_a_model_that_raises_becomes_a_recorded_failure():
    """Not an exception: 'this configuration is unusable' is a real data point,
    and the optimizer should keep searching."""
    scores, run_info = _run(Broken())

    assert scores == {name: 0.0 for name in METRICS}
    assert run_info["status"] == STATUS_CRASHED
    assert "this configuration explodes" in run_info["additional_info"]["error"]


def test_unscoreable_predictions_are_a_failure_not_a_crash():
    """A model can also fail by returning something wrong rather than raising."""
    scores, run_info = _run(WrongLength())

    assert scores == {name: 0.0 for name in METRICS}
    assert run_info["status"] == STATUS_CRASHED
    assert "could not be scored" in run_info["additional_info"]["error"]


def test_every_requested_metric_is_present_even_on_failure():
    """The optimizers index scores by the primary metric straight afterwards."""
    scores, _ = _run(Broken())
    assert set(scores) == set(METRICS)


def test_a_model_that_timed_itself_overrides_the_cpu_reading():
    """This thread's CPU clock measures nothing when the work happened in
    another process, so a model that reports its own is believed."""
    model = Perfect()
    model.last_cpu_time = 42.0

    _, run_info = _run(model)

    assert run_info["cpu_time"] == 42.0
