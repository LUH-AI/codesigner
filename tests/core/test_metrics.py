"""Scoring is the application's job.

A model returns predictions and never sees the validation labels, so these are
the only place a score is computed — which is what makes two models comparable
even when they run in different environments with different library versions.
"""

import numpy as np
import pytest

from core.metrics import METRICS, resolve, score_all


#: The names that were here before metrics could describe themselves. They go
#: into .ihpo files verbatim, so renaming one breaks every experiment that used
#: it — and changing what one *means* is worse, because nothing would say so.
LEGACY = ("accuracy", "f1", "precision", "recall(macro)")


def test_the_original_metric_names_still_mean_what_they_meant():
    """Presence, not set-equality: registering a new metric is a feature, and a
    test that fails for it would only ever be edited to say the new number.
    What must not change is these four — every stored score, every stored cost
    and every `1 - score` conversion in a file written before today assumes
    exactly this."""
    for name in LEGACY:
        metric = METRICS[name]
        assert metric.higher_is_better is True, name
        assert metric.bounds == (0.0, 1.0), name


def test_a_perfect_prediction_scores_one():
    y = np.array(["a", "b", "a", "b"], dtype=object)
    assert score_all(y, list(y), METRICS) == {name: 1.0 for name in METRICS}


def test_scores_are_plain_floats():
    """sklearn returns numpy scalars, which used to travel all the way into
    TrialResult.scores and rely on the serializer to coerce them."""
    y = np.array([0, 1, 0, 1])
    for value in score_all(y, [0, 1, 1, 1], METRICS).values():
        assert type(value) is float


def test_predictions_may_be_a_plain_list_against_object_labels():
    """That is how they come back from a model in another process."""
    y = np.array(["setosa", "virginica", "setosa"], dtype=object)
    assert score_all(y, ["setosa", "setosa", "setosa"], METRICS)["accuracy"] == pytest.approx(2 / 3)


def test_resolve_picks_named_metrics():
    picked = resolve(["accuracy", "f1"])
    assert set(picked) == {"accuracy", "f1"}


def test_resolve_names_the_ones_that_do_not_exist():
    with pytest.raises(ValueError, match="nope"):
        resolve(["accuracy", "nope"])


# ── what a metric knows about itself ─────────────────────────────────────────

def test_a_metric_says_which_direction_is_better():
    from core.metrics import Metric

    higher = METRICS["accuracy"]
    lower = Metric(name="rmse", fn=lambda y, yp: 0.0,
                   higher_is_better=False, bounds=(0.0, None))

    assert higher.better(0.9, 0.8) and not higher.better(0.8, 0.9)
    assert lower.better(0.8, 0.9) and not lower.better(0.9, 0.8)
    assert higher.best([0.2, 0.9, 0.5]) == 0.9
    assert lower.best([0.2, 0.9, 0.5]) == 0.2


def test_the_worst_sentinel_is_beaten_by_every_real_score():
    """It is what `float("-inf")` was, back when everything was a maximisation."""
    from core.metrics import Metric

    lower = Metric(name="rmse", fn=lambda y, yp: 0.0,
                   higher_is_better=False, bounds=(0.0, None))

    assert METRICS["accuracy"].better(0.0, METRICS["accuracy"].worst_sentinel)
    assert lower.better(1e9, lower.worst_sentinel)


# ── the cost an optimizer minimises ──────────────────────────────────────────

def test_the_original_metrics_still_cost_exactly_one_minus_the_score():
    """The whole point of deriving the conversion rather than branching on a
    flag: every `.ihpo` ever written keeps the cost it has, and no file needs a
    format bump to say which convention it used."""
    from core.metrics import to_cost

    for name in LEGACY:
        for score in (0.0, 0.25, 0.5, 1.0):
            assert to_cost(METRICS[name], score) == 1.0 - score


def test_a_metric_that_is_already_a_cost_round_trips_untouched():
    """An imported SMAC objective. SMAC minimises it and so do we, so there is
    nothing to convert — which is what makes importing one honest."""
    from core.metrics import Metric, from_cost, to_cost

    cost = Metric(name="cost", fn=lambda y, yp: 0.0,
                  higher_is_better=False, bounds=(None, None))

    for value in (-3.5, 0.0, 42.7, 1e6):
        assert to_cost(cost, value) == value
        assert from_cost(cost, value) == value


def test_an_unbounded_higher_is_better_metric_is_negated():
    """Nothing to subtract from, so negation is the only order-reversing map
    left — and it is exact, which a shift by an arbitrary constant would not be."""
    from core.metrics import Metric, from_cost, to_cost

    m = Metric(name="gain", fn=lambda y, yp: 0.0, bounds=(0.0, None))

    assert to_cost(m, 3.5) == -3.5
    assert from_cost(m, -3.5) == 3.5


def test_cost_and_score_are_exact_inverses_for_every_shape():
    from core.metrics import Metric, from_cost, to_cost

    shapes = [
        METRICS["accuracy"],
        Metric(name="rmse", fn=lambda *_: 0.0, higher_is_better=False, bounds=(0.0, None)),
        Metric(name="cost", fn=lambda *_: 0.0, higher_is_better=False, bounds=(None, None)),
        Metric(name="gain", fn=lambda *_: 0.0, bounds=(0.0, None)),
        None,   # a name this build has never heard of
    ]
    for metric in shapes:
        for score in (-2.0, 0.0, 0.375, 17.0):
            assert from_cost(metric, to_cost(metric, score)) == pytest.approx(score)


def test_a_lower_is_better_cost_orders_the_way_the_optimizer_needs():
    """The property that actually matters: whatever the metric, a better score
    must produce a smaller cost, because that is all SMAC is told."""
    from core.metrics import Metric, to_cost

    for metric in (METRICS["accuracy"],
                   Metric(name="rmse", fn=lambda *_: 0.0,
                          higher_is_better=False, bounds=(0.0, None))):
        better, worse = (0.9, 0.1) if metric.higher_is_better else (0.1, 0.9)
        assert to_cost(metric, better) < to_cost(metric, worse)


# ── scoring against probabilities ────────────────────────────────────────────

def test_a_probability_metric_is_handed_the_matrix_and_the_class_order():
    from core.metrics import LABELS, PROBABILITIES, Metric

    seen = {}

    def scorer(y_true, y_proba, classes):
        seen.update(proba=y_proba, classes=classes)
        return 0.75

    metrics = {"auc": Metric(name="auc", fn=scorer, needs=PROBABILITIES),
               "accuracy": METRICS["accuracy"]}
    y = np.array(["a", "b"], dtype=object)

    scores = score_all(y, ["a", "b"], metrics,
                       y_proba=[[0.9, 0.1], [0.2, 0.8]], classes=["a", "b"])

    assert scores == {"auc": 0.75, "accuracy": 1.0}
    assert seen["classes"] == ["a", "b"]
    assert METRICS["accuracy"].needs == LABELS


def test_asking_for_a_probability_metric_without_probabilities_says_so():
    """It cannot happen on a real run — `usable()` settles what can be scored
    before the first trial — so this is a programming error and reads as one."""
    from core.metrics import PROBABILITIES, Metric

    metrics = {"auc": Metric(name="auc", fn=lambda *_: 0.0, needs=PROBABILITIES)}

    with pytest.raises(ValueError, match="probabilities"):
        score_all(np.array(["a"]), ["a"], metrics)
