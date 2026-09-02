"""The explanation games are told which way the metric runs.

*What:* `HP_GAMES` map MAX/VAR/MIN onto tunability/sensitivity/mistunability,
and those names hold only when bigger is better. Handed an optimizer cost — what
a run imported from somebody else's output carries — MAX finds the configuration
that performed *worst* and the page labels it the most tunable. So the training
data handed to the explainer is turned, not the games.

*How:* `_pair_trials_with_oriented_scores` is checked directly against both
directions, and then against `fit_surrogate`, which must **not** be turned: it
feeds partial dependence, whose axis is labelled with the metric and has to stay
in the metric's own units.
"""

import pytest

from core.metrics import METRICS, Metric
from core.optimizers.base import (
    TrialResult, _pair_trials_with_oriented_scores, _pair_trials_with_scores,
)

COST = Metric(name="smac:cost", fn=None, higher_is_better=False, bounds=(None, None))


@pytest.fixture
def config_space():
    from ConfigSpace import ConfigurationSpace, Float, Integer

    return ConfigurationSpace(
        space={"depth": Integer("depth", (1, 10)),
               "rate": Float("rate", (0.01, 1.0))},
        seed=0)


def _trials(metric_name, scores):
    return [
        TrialResult(trial=i + 1, config={"depth": i + 1, "rate": 0.1 * (i + 1)},
                    scores={metric_name: s}, score=s, incumbent_score=s,
                    incumbent_config={}, run_info={}, origin="")
        for i, s in enumerate(scores)
    ]


def test_a_higher_is_better_metric_is_left_alone(config_space):
    """Every metric this build ships is one, so nothing about them moves."""
    trials = _trials("accuracy", [0.2, 0.9, 0.5])

    raw = _pair_trials_with_scores(config_space, trials, "accuracy")
    oriented = _pair_trials_with_oriented_scores(
        config_space, trials, "accuracy", METRICS["accuracy"])

    assert [s for _, s in oriented] == [s for _, s in raw]


def test_a_cost_is_negated(config_space):
    """Negation, so the best trial carries the largest number.

    Not `high - score`: an imported objective has no upper bound to subtract
    from. Both preserve every difference between two scores, which is what the
    games are built out of.
    """
    trials = _trials("smac:cost", [0.2, 0.9, 0.5])

    oriented = _pair_trials_with_oriented_scores(
        config_space, trials, "smac:cost", COST)

    assert [s for _, s in oriented] == [-0.2, -0.9, -0.5]
    assert max(s for _, s in oriented) == -0.2  # the cheapest trial


def test_differences_survive_the_turn(config_space):
    """The games read gaps between scores, and the gaps are unchanged."""
    trials = _trials("smac:cost", [0.2, 0.9])

    (_, a), (_, b) = _pair_trials_with_oriented_scores(
        config_space, trials, "smac:cost", COST)

    assert abs(b - a) == pytest.approx(0.7)


def test_the_registry_answers_when_no_metric_is_handed_over(config_space):
    """Every call site that predates this passes nothing and is unaffected."""
    trials = _trials("accuracy", [0.2, 0.9, 0.5])

    oriented = _pair_trials_with_oriented_scores(config_space, trials, "accuracy")

    assert [s for _, s in oriented] == [0.2, 0.9, 0.5]


def test_the_partial_dependence_surrogate_is_not_turned(config_space):
    """`fit_surrogate` stays in the metric's own units, deliberately.

    Partial dependence labels its y-axis with the metric, and the uncertainty
    field's spread across trees is the same either way — so a surrogate fitted
    on negated costs would draw a PDP whose axis said "smac:cost" and whose
    values were its negation.
    """
    from core.optimizers.base import fit_surrogate

    trials = _trials("smac:cost", [0.2, 0.9, 0.5, 0.4])

    surrogate, warning = fit_surrogate(config_space, trials, "smac:cost", seed=0)

    assert warning is None
    # Predictions sit inside the raw cost range, not its negation.
    predictions = surrogate.predict([c.get_array() for c, _ in
                                     _pair_trials_with_scores(config_space, trials,
                                                              "smac:cost")])
    assert min(predictions) >= 0.0
