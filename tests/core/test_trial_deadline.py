"""`core.modelhost.deadline` policies set each trial's time limit, and
`PredictedDeadline` scales it per configuration. A run survives a timeout with a
fresh child. The in-process path enforces no deadline.
"""

import sys
import time

import pytest

from core.modelhost import model_session
from core.modelhost.deadline import (
    DEFAULT_TRIAL_TIMEOUT, MIN_OBSERVATIONS, MIN_PREDICTED_DEADLINE, NO_LIMIT, FixedDeadline,
    PredictedDeadline, as_deadline,
)
from core.modelhost.errors import TrialTimeout
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS, STATUS_TIMEOUT
from core.optimizers.trial import evaluate_trial

from tests.conftest import launch_local

from .test_modelhost import METRICS, SPLITS, _model, _session


def test_a_bare_number_still_means_what_it_always_did():
    assert as_deadline(30.0).seconds_for({"k": 1}) == 30.0
    assert as_deadline(None).seconds_for({"k": 1}) == DEFAULT_TRIAL_TIMEOUT


def test_zero_seconds_is_no_limit_rather_than_no_time():
    """`FixedDeadline(0)` means no limit, like other numeric fields on the run
    form."""
    assert FixedDeadline(0).seconds_for({"k": 1}) == NO_LIMIT
    assert as_deadline({"mode": "fixed", "seconds": 0}).seconds_for({"k": 1}) == NO_LIMIT


def test_an_unreadable_spec_falls_back_to_fixed_rather_than_raising():
    """Older JSON columns lack the field."""
    assert isinstance(as_deadline({}), FixedDeadline)
    assert isinstance(as_deadline({"mode": "nonsense"}), FixedDeadline)
    assert isinstance(as_deadline("junk"), FixedDeadline)
    assert as_deadline({"mode": "fixed", "seconds": "abc"}).seconds_for({}) == DEFAULT_TRIAL_TIMEOUT


def test_a_predicted_deadline_is_the_ceiling_until_it_has_seen_enough():
    """`PredictedDeadline` returns the ceiling before `MIN_OBSERVATIONS` trials."""
    policy = PredictedDeadline(factor=3.0, ceiling=100.0)
    policy.bind(_space())

    assert policy.seconds_for({"k": 1}) == 100.0
    for _ in range(MIN_OBSERVATIONS - 1):
        policy.observe({"k": 1}, 0.5, completed=True)
    assert policy.seconds_for({"k": 1}) == 100.0, "one short is still the ceiling"


def test_a_slow_configuration_is_given_longer_than_a_fast_one():
    """`PredictedDeadline` fits the deadline to the configuration, which a
    median-relative cap cannot."""
    policy = PredictedDeadline(factor=2.0, ceiling=10_000.0, floor=0.0)
    policy.bind(_space())

    # k is the only hyperparameter. Eight rows per value keep bootstrap noise
    # from hiding the k trend.
    for _ in range(8):
        for k in (1, 5):
            policy.observe({"k": k}, 1.0 if k == 1 else 100.0, completed=True)

    fast, slow = policy.seconds_for({"k": 1}), policy.seconds_for({"k": 5})
    assert slow > fast * 10, f"expected the big configuration to get far longer, got {fast=} {slow=}"
    assert fast == pytest.approx(2.0, rel=0.5), "roughly twice its own predicted duration"


def test_the_floor_keeps_a_fast_prediction_from_becoming_no_time_at_all():
    """`MIN_PREDICTED_DEADLINE` keeps the deadline above the round trip itself."""
    policy = PredictedDeadline(factor=2.0, ceiling=10_000.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 0.05, completed=True)

    assert policy.seconds_for({"k": 1}) == MIN_PREDICTED_DEADLINE


def test_the_ceiling_is_never_exceeded_however_slow_the_prediction():
    """The ceiling bounds the forest, which cannot extrapolate past its training
    durations."""
    policy = PredictedDeadline(factor=100.0, ceiling=60.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 3}, 30.0, completed=True)

    assert policy.seconds_for({"k": 3}) == 60.0


def test_a_timeout_is_not_learned_from():
    """`observe` skips timeouts, since learning from one would teach the forest
    a slow region is fast."""
    policy = PredictedDeadline(factor=2.0, ceiling=500.0, floor=0.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 1.0, completed=False)

    assert policy.seconds_for({"k": 1}) == 500.0, "nothing was learned, so the ceiling stands"


def test_an_unbound_policy_stays_on_its_ceiling_rather_than_failing():
    policy = PredictedDeadline(factor=2.0, ceiling=42.0)
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 0.1, completed=True)
    assert policy.seconds_for({"k": 1}) == 42.0


def _space():
    from ConfigSpace import ConfigurationSpace, Integer

    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("k", (1, 5), default=3)])
    return cs


def test_a_timeout_is_recorded_as_a_timeout_and_not_as_a_crash(tmp_path):
    """A crash judges the configuration, while a timeout leaves the outcome
    unknown."""
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")

    with _session(model, trial_timeout=1.0) as remote:
        scores, run_info = evaluate_trial(remote, {"k": 1}, SPLITS, METRICS, seed=0)

    assert run_info["status"] == STATUS_TIMEOUT
    assert run_info["status"] != STATUS_CRASHED
    assert scores == {name: 0.0 for name in METRICS}
    assert "traceback" not in run_info["additional_info"], (
        "the deadline expired over here, so a traceback would show this loop")


def test_the_run_survives_a_timeout(tmp_path):
    """`model_session` replaces the killed child, so the next trial measures
    instead of crashing on a dead pipe.
    """
    model = _model(tmp_path, """
        import time
        if config["k"] == 1:
            time.sleep(30)
        return ['a', 'b']
    """)

    with _session(model, trial_timeout=1.0) as remote:
        _, timed_out = evaluate_trial(remote, {"k": 1}, SPLITS, METRICS, seed=0)
        _, after = evaluate_trial(remote, {"k": 2}, SPLITS, METRICS, seed=0)
        assert remote._process.alive, "the replacement child should be running"

    assert timed_out["status"] == STATUS_TIMEOUT
    assert after["status"] == STATUS_SUCCESS, (
        "the trial after a timeout is a real measurement, not a fabricated crash")


def test_a_cancelled_run_is_not_given_a_replacement_child(tmp_path):
    """`start()` has no cancel flag, and a replacement child would wait out the
    whole start timeout."""
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")
    flag = _Cancel()

    with _session(model, cancel=flag, trial_timeout=1.0) as remote:
        with pytest.raises(TrialTimeout):
            remote.fit_predict({"k": 1}, None, None, None)
        assert not remote._process.alive

        flag.value = True
        started = time.monotonic()
        with pytest.raises(Exception) as caught:
            remote.fit_predict({"k": 2}, None, None, None)
        assert time.monotonic() - started < 5.0, "should refuse, not spawn and wait"
        assert "cancelled" in str(caught.value)


def test_the_deadline_follows_the_configuration_against_a_real_child(tmp_path):
    model = _model(tmp_path, """
        import time
        time.sleep(2.0 if config["k"] == 5 else 0.01)
        return ['a', 'b']
    """)
    policy = PredictedDeadline(factor=2.0, ceiling=60.0, floor=0.5)

    with _session(model, trial_timeout=policy) as remote:
        for _ in range(MIN_OBSERVATIONS + 1):
            remote.fit_predict({"k": 1}, None, None, None)
        # The policy fits on ~0.01s calls, and the floor keeps fast trials from
        # timing out.
        assert policy.seconds_for({"k": 1}) == 0.5
        with pytest.raises(TrialTimeout):
            remote.fit_predict({"k": 5}, None, None, None)


class _Cancel:
    def __init__(self):
        self.value = False

    def is_set(self) -> bool:
        return self.value


def test_an_in_process_model_is_not_subject_to_any_deadline():
    """Python cannot interrupt the run's own thread, where a registry model's
    `fit_predict` runs.
    """
    class SlowLocalModel:
        name = "Slow"

        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            time.sleep(0.4)
            return ["a", "b"]

    started = time.monotonic()
    scores, run_info = evaluate_trial(
        SlowLocalModel(), {"k": 1}, SPLITS, METRICS, seed=0)
    elapsed = time.monotonic() - started

    assert run_info["status"] == STATUS_SUCCESS
    assert elapsed >= 0.4, (
        "nothing interrupted it — there is no deadline on the in-process path")
    assert scores


def test_the_local_path_takes_no_timeout_argument_at_all():
    """`experiments/services/run.py` calls `optimize` directly on the in-process
    path, with no `model_session` to own a deadline."""
    import inspect

    from core.optimizers.base import BaseOptimizer

    assert "trial_timeout" not in inspect.signature(BaseOptimizer.optimize).parameters
    assert "trial_timeout" in inspect.signature(model_session.__init__).parameters


def test_a_session_accepts_a_policy_where_it_used_to_take_a_number(tmp_path):
    model = _model(tmp_path, "return ['a', 'b']")
    launch = launch_local(sys.executable, model)

    with model_session(launch, SPLITS, trial_timeout=FixedDeadline(5.0)) as remote:
        assert remote._deadline.seconds_for({"k": 1}) == 5.0
    with model_session(launch, SPLITS, trial_timeout=7.0) as remote:
        assert remote._deadline.seconds_for({"k": 1}) == 7.0
