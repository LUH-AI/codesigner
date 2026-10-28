"""`SMACOptimizer.evaluate_prior` judges priors here on a made-up objective
whose optimum is known.
"""

import random

import pytest
from ConfigSpace import ConfigurationSpace, Float

from core.optimizers import SMACOptimizer
from core.optimizers.base import TrialResult

pytest.importorskip(
    "smac.acquisition.weight", reason="needs the SMAC branch carrying the climbing comparison policy")
from smac.acquisition.weight import ClimbingComparisonPolicy  # noqa: E402

#: Peak of the made-up objective along `x0`.
OPTIMUM = 0.7


def _trials(dims, n=30):
    """`_trials` builds *n* trials whose accuracy peaks at `x0` = OPTIMUM."""
    space = ConfigurationSpace(seed=0)
    space.add([Float(f"x{i}", (0.0, 1.0)) for i in range(dims)])
    noise = random.Random(3)
    trials = []
    for i, config in enumerate(space.sample_configuration(n)):
        values = dict(config)
        score = (0.9 - 2.0 * (values["x0"] - OPTIMUM) ** 2
                 - sum(0.3 * (values[f"x{k}"] - 0.4) ** 2 for k in range(1, dims))
                 + noise.gauss(0, 0.005))
        trials.append(TrialResult(trial=i, config=values, scores={"accuracy": score},
                                  score=score, incumbent_score=score, incumbent_config=values))
    return space, trials


def _belief(mu):
    return {"x0": {"kind": "normal", "params": {"mu": mu, "sigma": 0.05},
                   "decay": {"shape": "none"}, "at_trial": 0}}


def _judge(space, trials, mu, **kwargs):
    return SMACOptimizer(search_strategy="gp").evaluate_prior(
        space, trials, "accuracy", 0, priors=_belief(mu), **kwargs)


@pytest.mark.parametrize("dims", [2, 4])
def test_a_belief_at_the_optimum_is_accepted(dims):
    """A correct belief about `x0` passes however many other hyperparameters it
    leaves open."""
    space, trials = _trials(dims)

    assert _judge(space, trials, OPTIMUM)["verdict"] == "accepted"


@pytest.mark.parametrize("dims", [2, 4])
def test_a_belief_far_from_the_optimum_is_refused(dims):
    space, trials = _trials(dims)

    assert _judge(space, trials, 0.05)["verdict"] == "rejected"


def test_top_k_reaches_the_policy(monkeypatch):
    """`evaluate_prior` passes top k, per-hyperparameter sampling and a
    prior-wide neighbourhood to `ClimbingComparisonPolicy`."""
    seen = {}
    original = ClimbingComparisonPolicy.__init__

    def _recording(self, **kwargs):
        seen.update(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(ClimbingComparisonPolicy, "__init__", _recording)
    space, trials = _trials(2)
    _judge(space, trials, OPTIMUM, top_k=3, tolerance=0.2)

    assert seen["top_k"] == 3
    assert seen["threshold"] == pytest.approx(-0.2)
    assert seen["n_samples_per_hyperparameter"] > 0
    assert seen["neighbourhood_std"] == "prior"


def test_a_check_that_compared_nothing_is_not_an_approval(monkeypatch):
    """`evaluate_prior` reports "unjudged" when the policy comparison finds
    nothing."""
    monkeypatch.setattr(ClimbingComparisonPolicy, "compare", lambda self, *a, **k: None)
    space, trials = _trials(2)

    assert _judge(space, trials, OPTIMUM)["verdict"] == "unjudged"


def test_no_trials_is_not_judged():
    space, _trials_unused = _trials(2)

    assert _judge(space, [], OPTIMUM)["verdict"] == "unjudged"
