"""Judging a stated prior against the model, as "Evaluate Prior" does.

`SMACOptimizer.evaluate_prior` rebuilds the facade, replays the trials and
offers the prior to `add_prior` behind SMAC's `ClimbingComparisonPolicy`. These
run it on a made-up objective whose optimum is known, so a good and a bad
belief can be told apart without a stored run.
"""

import random

import pytest
from ConfigSpace import ConfigurationSpace, Float

from core.optimizers import SMACOptimizer
from core.optimizers.base import TrialResult

pytest.importorskip(
    "smac.acquisition.weight", reason="needs the SMAC branch carrying the climbing comparison policy")
from smac.acquisition.weight import ClimbingComparisonPolicy  # noqa: E402

#: Where the made-up objective peaks along `x0`, the hyperparameter the priors
#: here are about. The others peak at 0.4 and matter less.
OPTIMUM = 0.7


def _trials(dims, n=30):
    """*n* trials of an accuracy that peaks at `x0` = OPTIMUM, over *dims*
    hyperparameters, so a belief about `x0` alone leaves the rest open."""
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
    """*What:* a correct belief about one hyperparameter passes, however many
    others it leaves open — the case the old mean comparison refused.

    *How:* a Normal at the made-up objective's optimum, over two and four
    hyperparameters."""
    space, trials = _trials(dims)

    assert _judge(space, trials, OPTIMUM)["verdict"] == "accepted"


@pytest.mark.parametrize("dims", [2, 4])
def test_a_belief_far_from_the_optimum_is_refused(dims):
    """*What:* a belief pointing where the objective is far worse is refused.

    *How:* a Normal at 0.05, where the made-up accuracy is about 0.85 below
    its peak — far past the default tolerance."""
    space, trials = _trials(dims)

    assert _judge(space, trials, 0.05)["verdict"] == "rejected"


def test_top_k_reaches_the_policy(monkeypatch):
    """*What:* the top k asked for is the one SMAC climbs from, the policy
    draws per hyperparameter rather than a fixed count, and the incumbent's
    neighbourhood is as wide as the belief, as DynaBO draws it.

    *How:* the policy's constructor wrapped to record what it was given."""
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
    """*What:* when the policy has nothing to compare it accepts, as SMAC
    requires, and that must reach the page as "unjudged", not "accepted".

    *How:* the policy's comparison made to find nothing, as it does with no
    model or no incumbent."""
    monkeypatch.setattr(ClimbingComparisonPolicy, "compare", lambda self, *a, **k: None)
    space, trials = _trials(2)

    assert _judge(space, trials, OPTIMUM)["verdict"] == "unjudged"


def test_no_trials_is_not_judged():
    """*What:* before any trial there is no model, so nothing is judged.

    *How:* the same belief offered with an empty history."""
    space, _trials_unused = _trials(2)

    assert _judge(space, [], OPTIMUM)["verdict"] == "unjudged"
