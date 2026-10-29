"""`RemoteModel` raises `TrialCancelled` out of `fit_predict` when cancellation
reaches a running trial, and the interrupted trial has no score.
`evaluate_trial` returns zeros to keep its shape, and `TrialCollector.record`
drops the attempt. A recorded zero would skew the surrogates and count against
`max_failures`. The collector implements the rule because all three optimizers call
`record`.
"""

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Integer

from core.modelhost.errors import TrialCancelled
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer
from core.optimizers.base import STOPPED_BY_CANCELLED, TrialCollector
from core.optimizers.timing import CANCELLED, STATUS_CRASHED, STATUS_SUCCESS
from core.optimizers.trial import evaluate_trial


class _Flag:
    """`_Flag` mimics a cancel flag that trips mid-run."""

    def __init__(self):
        self.value = False

    def is_set(self) -> bool:
        return self.value


class CancelsPartWay:
    """`CancelsPartWay` mimics `RemoteModel.fit_predict` raising
    `TrialCancelled` without a subprocess.
    """

    name = "Cancels Part Way"

    def __init__(self, flag, after: int = 2):
        self.flag = flag
        self.after = after
        self.calls = 0

    def get_config_space(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("a", (1, 10), default=5)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        self.calls += 1
        if self.calls > self.after:
            self.flag.value = True
            raise TrialCancelled("the run was cancelled")
        labels = sorted({str(label) for label in y_train})
        return [labels[i % len(labels)] for i in range(len(X_val))]


@pytest.fixture
def data():
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = np.array(["a", "b"] * 30, dtype=object)
    return X, y


def test_evaluate_trial_marks_a_cancellation_rather_than_calling_it_a_failure(data, metrics):
    """`CANCELLED` is a marker beside the status because SMAC's `StatusType`
    lacks a cancelled member."""
    from core.splits import holdout

    X, y = data
    flag = _Flag()
    flag.value = True
    model = CancelsPartWay(flag, after=0)

    scores, run_info = evaluate_trial(
        model, {"a": 1}, holdout(X, y, X, y), metrics, seed=0)

    assert run_info[CANCELLED] is True
    assert CANCELLED not in {"status"}, "it is a marker beside the status, not one of them"
    assert scores == {name: 0.0 for name in metrics}, "shape kept; never recorded"


def test_the_collector_records_nothing_for_a_cancelled_attempt(metrics):
    """`TrialCollector` gates all three optimizers."""
    collector = TrialCollector(stopping={"max_trials": 10})
    collector.record({"a": 1}, 0.8, {"accuracy": 0.8}, run_info={"status": STATUS_SUCCESS})

    dropped = collector.record({"a": 2}, 0.0, {"accuracy": 0.0},
                               run_info={"status": STATUS_CRASHED, CANCELLED: True})

    assert dropped is None
    assert len(collector.results) == 1
    assert [t.config for t in collector.results] == [{"a": 1}]


def test_a_cancelled_attempt_does_not_count_as_a_failure():
    """With `max_failures` at its default of 1, one recorded crash would end the
    run as failing."""
    collector = TrialCollector(stopping={"max_trials": 10})
    collector.record({"a": 1}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED, CANCELLED: True})

    assert collector._failures == 0
    assert collector._consecutive_failures == 0
    assert collector.stopped_by == STOPPED_BY_CANCELLED


def test_a_cancelled_attempt_does_not_move_the_incumbent_or_the_clock():
    """`TrialCollector` feeds stopping rules from `_since_improvement` and
    `_trial_seconds`."""
    collector = TrialCollector(stopping={"max_trials": 10})
    collector.record({"a": 1}, 0.8, {"accuracy": 0.8},
                     run_info={"status": STATUS_SUCCESS, "time": 1.0})
    before = (collector.incumbent_score, collector._since_improvement,
              collector._trial_seconds)

    collector.record({"a": 2}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED, CANCELLED: True, "time": 5.0})

    assert (collector.incumbent_score, collector._since_improvement,
            collector._trial_seconds) == before


def test_the_collector_ends_the_run_on_its_own_terms():
    """`done` latches, and loops need no second flag check."""
    collector = TrialCollector(stopping={"max_trials": 100})
    assert not collector.done

    collector.record({"a": 1}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED, CANCELLED: True})

    assert collector.done
    assert collector.stopped_by == STOPPED_BY_CANCELLED


def _optimizers():
    return [
        pytest.param(RandomOptimizer(), id="random"),
        pytest.param(GridOptimizer(), id="grid"),
        pytest.param(SMACOptimizer(), id="smac", marks=pytest.mark.slow),
    ]


@pytest.mark.parametrize("optimizer", _optimizers())
def test_a_cancelled_run_keeps_its_measured_trials_and_only_those(optimizer, data, metrics):
    """None of the three loops has cancellation code beyond the flag check; the
    collector decides.
    """
    X, y = data
    flag = _Flag()
    model = CancelsPartWay(flag, after=2)

    result = optimizer.optimize(
        model, X, y, X, y, metrics=metrics, primary_metric="accuracy",
        n_trials=10, seed=0, cancel_event=flag)

    assert len(result.trials) == 2, "the interrupted attempt is not a trial"
    assert [t.trial for t in result.trials] == [1, 2], "and leaves no gap in the numbering"
    for t in result.trials:
        assert t.run_info.get("status") == STATUS_SUCCESS
        assert CANCELLED not in t.run_info


@pytest.mark.parametrize("optimizer", _optimizers())
def test_a_cancelled_run_says_it_was_interrupted(optimizer, data, metrics):
    X, y = data
    flag = _Flag()

    result = optimizer.optimize(
        CancelsPartWay(flag, after=2), X, y, X, y,
        metrics=metrics, primary_metric="accuracy",
        n_trials=10, seed=0, cancel_event=flag)

    assert result.metadata.get("stopped_by") == STOPPED_BY_CANCELLED


@pytest.mark.parametrize("optimizer", _optimizers())
def test_no_zero_scoring_trial_reaches_the_stored_file(optimizer, data, metrics):
    """Nothing unmeasured reaches the stored file."""
    X, y = data
    flag = _Flag()

    result = optimizer.optimize(
        CancelsPartWay(flag, after=3), X, y, X, y,
        metrics=metrics, primary_metric="accuracy",
        n_trials=10, seed=0, cancel_event=flag)

    stored = optimizer.serialize_result(result)
    assert stored["data"], "the trials that did run are still saved"
    for entry in stored["data"]:
        assert entry["status"] == STATUS_SUCCESS
        assert entry["scores"]["accuracy"] > 0.0
