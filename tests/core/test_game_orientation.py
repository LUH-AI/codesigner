"""`HP_GAMES` names only apply when bigger is better, and
`pair_trials_with_oriented_scores` negates a cost before the games see the
scores. `fit_surrogate` stays in the metric's units because partial dependence
plots them.
"""

import pytest

from core.metrics import METRICS, Metric, unscoreable
from core.analytics import pair_trials_with_scores
from core.analytics.surrogate import pair_trials_with_oriented_scores
from core.optimizers.base import TrialResult

COST = Metric(name="smac:cost", fn=unscoreable, higher_is_better=False, bounds=(None, None))


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
    trials = _trials("accuracy", [0.2, 0.9, 0.5])

    raw = pair_trials_with_scores(config_space, trials, "accuracy")
    oriented = pair_trials_with_oriented_scores(
        config_space, trials, "accuracy", METRICS["accuracy"])

    assert [s for _, s in oriented] == [s for _, s in raw]


def test_a_cost_is_negated(config_space):
    """Negation needs no upper bound, which an imported objective lacks.
    """
    trials = _trials("smac:cost", [0.2, 0.9, 0.5])

    oriented = pair_trials_with_oriented_scores(
        config_space, trials, "smac:cost", COST)

    assert [s for _, s in oriented] == [-0.2, -0.9, -0.5]
    assert max(s for _, s in oriented) == -0.2  # cheapest trial


def test_differences_survive_the_turn(config_space):
    """The games read gaps, which stay unchanged."""
    trials = _trials("smac:cost", [0.2, 0.9])

    (_, a), (_, b) = pair_trials_with_oriented_scores(
        config_space, trials, "smac:cost", COST)

    assert abs(b - a) == pytest.approx(0.7)


def test_the_registry_answers_when_no_metric_is_handed_over(config_space):
    """Older call sites pass no metric."""
    trials = _trials("accuracy", [0.2, 0.9, 0.5])

    oriented = pair_trials_with_oriented_scores(config_space, trials, "accuracy")

    assert [s for _, s in oriented] == [0.2, 0.9, 0.5]


def test_the_partial_dependence_surrogate_is_not_turned(config_space):
    """Negated predictions would contradict the metric label on the partial
    dependence axis.
    """
    from core.analytics import fit_surrogate

    trials = _trials("smac:cost", [0.2, 0.9, 0.5, 0.4])

    surrogate, warning = fit_surrogate(config_space, trials, "smac:cost", seed=0)

    assert warning is None
    assert surrogate is not None
    # Predictions stay inside the raw cost range.
    predictions = surrogate.predict([c.get_array() for c, _ in
                                     pair_trials_with_scores(config_space, trials,
                                                              "smac:cost")])
    assert min(predictions) >= 0.0
