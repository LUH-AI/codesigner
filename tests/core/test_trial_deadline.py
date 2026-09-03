"""How long a trial is given, and what happens to the run when it runs out.

Three things, in the order they matter:

- a deadline is *enforced* and recorded as a timeout rather than a crash;
- a run *survives* one, which it did not before — the child that missed its
  deadline has to be killed, and every trial after it used to find a dead pipe;
- the deadline can be *predicted per configuration* instead of fixed, so a
  large configuration is given proportionally longer rather than killed for
  being large.

And one thing that is a limitation rather than a feature: the in-process path
cannot enforce any of this, asserted here so it is documented by a test rather
than only by a comment.
"""

import sys
import time

import pytest

from core.modelhost import (
    DEFAULT_TRIAL_TIMEOUT,
    FixedDeadline,
    PredictedDeadline,
    TrialTimeout,
    as_deadline,
    launch_local,
    model_session,
)
from core.modelhost.deadline import (
    MIN_OBSERVATIONS,
    MIN_PREDICTED_DEADLINE,
    NO_LIMIT,
)
from core.optimizers.timing import STATUS_CRASHED, STATUS_SUCCESS, STATUS_TIMEOUT
from core.optimizers.trial import evaluate_trial

from .test_modelhost import METRICS, SPLITS, _model, _session


# ── the policies, on their own ───────────────────────────────────────────────

def test_a_bare_number_still_means_what_it_always_did():
    """Every existing caller passes a float. That has to keep working."""
    assert as_deadline(30.0).seconds_for({"k": 1}) == 30.0
    assert as_deadline(None).seconds_for({"k": 1}) == DEFAULT_TRIAL_TIMEOUT


def test_zero_seconds_is_no_limit_rather_than_no_time():
    """0 in the box reads as "no deadline", matching every other numeric field
    on the run form. The alternative reading — a deadline of zero — would kill
    every trial instantly, which nobody has ever meant by leaving a limit off."""
    assert FixedDeadline(0).seconds_for({"k": 1}) == NO_LIMIT
    assert as_deadline({"mode": "fixed", "seconds": 0}).seconds_for({"k": 1}) == NO_LIMIT


def test_an_unreadable_spec_falls_back_to_fixed_rather_than_raising():
    """Read out of a JSON column that predates the field."""
    assert isinstance(as_deadline({}), FixedDeadline)
    assert isinstance(as_deadline({"mode": "nonsense"}), FixedDeadline)
    assert isinstance(as_deadline("junk"), FixedDeadline)
    assert as_deadline({"mode": "fixed", "seconds": "abc"}).seconds_for({}) == DEFAULT_TRIAL_TIMEOUT


def test_a_predicted_deadline_is_the_ceiling_until_it_has_seen_enough():
    """The earliest trials are the ones there is least to go on for, so they get
    exactly the fixed deadline they would have had."""
    policy = PredictedDeadline(factor=3.0, ceiling=100.0)
    policy.bind(_space())

    assert policy.seconds_for({"k": 1}) == 100.0
    for _ in range(MIN_OBSERVATIONS - 1):
        policy.observe({"k": 1}, 0.5, completed=True)
    assert policy.seconds_for({"k": 1}) == 100.0, "one short is still the ceiling"


def test_a_slow_configuration_is_given_longer_than_a_fast_one():
    """The whole point, and the thing a median-relative cap cannot do: the
    deadline follows the configuration, not the run's typical trial."""
    policy = PredictedDeadline(factor=2.0, ceiling=10_000.0, floor=0.0)
    policy.bind(_space())

    # k is the only hyperparameter; make duration track it steeply. Eight of
    # each rather than the bare minimum: the forest bootstraps, so with only a
    # handful of rows some trees see one value of k and none of the other, and
    # what is under test here is the relationship, not the small-sample noise.
    for _ in range(8):
        for k in (1, 5):
            policy.observe({"k": k}, 1.0 if k == 1 else 100.0, completed=True)

    fast, slow = policy.seconds_for({"k": 1}), policy.seconds_for({"k": 5})
    assert slow > fast * 10, f"expected the big configuration to get far longer, got {fast=} {slow=}"
    assert fast == pytest.approx(2.0, rel=0.5), "roughly twice its own predicted duration"


def test_the_floor_keeps_a_fast_prediction_from_becoming_no_time_at_all():
    """A model measured at 0.05s would otherwise get a deadline shorter than the
    round trip that carries it."""
    policy = PredictedDeadline(factor=2.0, ceiling=10_000.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 0.05, completed=True)

    assert policy.seconds_for({"k": 1}) == MIN_PREDICTED_DEADLINE


def test_the_ceiling_is_never_exceeded_however_slow_the_prediction():
    """The fixed number stays the outer bound — it is what covers the forest's
    inability to extrapolate past the durations it was trained on."""
    policy = PredictedDeadline(factor=100.0, ceiling=60.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 3}, 30.0, completed=True)

    assert policy.seconds_for({"k": 3}) == 60.0


def test_a_timeout_is_not_learned_from():
    """Censored, not measured: all a timeout says is "longer than what it was
    given". Training on it would teach the forest the region is fast and tighten
    the deadline over it — the one feedback loop worth refusing."""
    policy = PredictedDeadline(factor=2.0, ceiling=500.0, floor=0.0)
    policy.bind(_space())
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 1.0, completed=False)

    assert policy.seconds_for({"k": 1}) == 500.0, "nothing was learned, so the ceiling stands"


def test_an_unbound_policy_stays_on_its_ceiling_rather_than_failing():
    """No search space to encode against — a safe way to be uninformed."""
    policy = PredictedDeadline(factor=2.0, ceiling=42.0)
    for _ in range(MIN_OBSERVATIONS * 2):
        policy.observe({"k": 1}, 0.1, completed=True)
    assert policy.seconds_for({"k": 1}) == 42.0


def _space():
    from ConfigSpace import ConfigurationSpace, Integer

    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("k", (1, 5), default=3)])
    return cs


# ── enforced against a real child ────────────────────────────────────────────

def test_a_timeout_is_recorded_as_a_timeout_and_not_as_a_crash(tmp_path):
    """`docs/plan_today.md` Step 2's own verification. The distinction is not
    cosmetic: a crash means the model had an opinion about the configuration, a
    timeout means nobody knows."""
    model = _model(tmp_path, "import time\ntime.sleep(30)\nreturn ['a', 'b']")

    with _session(model, trial_timeout=1.0) as remote:
        scores, run_info = evaluate_trial(remote, {"k": 1}, SPLITS, METRICS, seed=0)

    assert run_info["status"] == STATUS_TIMEOUT
    assert run_info["status"] != STATUS_CRASHED
    assert scores == {name: 0.0 for name in METRICS}
    assert "traceback" not in run_info["additional_info"], (
        "the deadline expired over here, so a traceback would show this loop")


def test_the_run_survives_a_timeout(tmp_path):
    """The regression this feature could not ship without.

    Enforcing a deadline means killing the child that missed it. Before the
    restart, that kill took the rest of the run with it: every later trial found
    a dead pipe and was recorded as a crash, so one slow configuration produced a
    run of fabricated failures. Now the next trial gets a fresh child and a real
    measurement.
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
    """`start()` has no cancel flag to read, so a child spawned for a run nobody
    wants would be waited on for the whole start timeout."""
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
    """End to end: the same session kills a slow configuration and lets a fast
    one through, under one policy whose numbers came from the fast ones."""
    model = _model(tmp_path, """
        import time
        time.sleep(2.0 if config["k"] == 5 else 0.01)
        return ['a', 'b']
    """)
    policy = PredictedDeadline(factor=2.0, ceiling=60.0, floor=0.5)

    with _session(model, trial_timeout=policy) as remote:
        for _ in range(MIN_OBSERVATIONS + 1):
            remote.fit_predict({"k": 1}, None, None, None)
        # Fitted entirely on ~0.01s calls, so anything much slower is refused —
        # and the floor is what keeps the fast ones from refusing themselves.
        assert policy.seconds_for({"k": 1}) == 0.5
        with pytest.raises(TrialTimeout):
            remote.fit_predict({"k": 5}, None, None, None)


class _Cancel:
    def __init__(self):
        self.value = False

    def is_set(self) -> bool:
        return self.value


# ── the limitation, documented by a test ─────────────────────────────────────

def test_an_in_process_model_is_not_subject_to_any_deadline():
    """`docs/plan_today.md` Step 2: a registry model's `fit_predict` runs in the
    run's own thread, and Python cannot interrupt that. No timeout is enforced
    there and none can be without running every trial out of process.

    Asserted rather than commented, because "the deadline did not apply" is
    exactly the kind of thing a reader assumes is a bug.
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
    assert scores  # and it produced a real measurement


def test_the_local_path_takes_no_timeout_argument_at_all():
    """The seam itself: a deadline is a `model_session` concept, and there is no
    session on the in-process path — `ui/services/run.py` calls `optimize`
    directly with the imported model."""
    import inspect

    from core.optimizers.base import BaseOptimizer

    assert "trial_timeout" not in inspect.signature(BaseOptimizer.optimize).parameters
    assert "trial_timeout" in inspect.signature(model_session.__init__).parameters


# ── the session accepts either shape ─────────────────────────────────────────

def test_a_session_accepts_a_policy_where_it_used_to_take_a_number(tmp_path):
    model = _model(tmp_path, "return ['a', 'b']")
    launch = launch_local(sys.executable, model)

    with model_session(launch, SPLITS, trial_timeout=FixedDeadline(5.0)) as remote:
        assert remote._deadline.seconds_for({"k": 1}) == 5.0
    with model_session(launch, SPLITS, trial_timeout=7.0) as remote:
        assert remote._deadline.seconds_for({"k": 1}) == 7.0
