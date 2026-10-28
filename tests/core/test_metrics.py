"""`core.metrics` alone computes scores, keeping models comparable across
environments.
"""

import numpy as np
import pytest

from core.metrics import metrics_for, resolve, score_all

#: Every split here has class labels only.
METRICS = resolve(metrics_for("classification"))


#: Renaming or redefining a `LEGACY` name breaks old `.ihpo` files.
LEGACY = ("accuracy", "f1", "precision", "recall(macro)")


def test_the_original_metric_names_still_mean_what_they_meant():
    """New metrics may register freely, and these four must stay fixed."""
    for name in LEGACY:
        metric = METRICS[name]
        assert metric.higher_is_better is True, name
        assert metric.bounds == (0.0, 1.0), name


def test_a_perfect_prediction_scores_one():
    y = np.array(["a", "b", "a", "b"], dtype=object)
    assert score_all(y, list(y), METRICS) == {name: 1.0 for name in METRICS}


def test_scores_are_plain_floats():
    """`score_all` converts sklearn's numpy scalars to plain floats."""
    y = np.array([0, 1, 0, 1])
    for value in score_all(y, [0, 1, 1, 1], METRICS).values():
        assert type(value) is float


def test_predictions_may_be_a_plain_list_against_object_labels():
    """Models in another process return plain lists."""
    y = np.array(["setosa", "virginica", "setosa"], dtype=object)
    assert score_all(y, ["setosa", "setosa", "setosa"], METRICS)["accuracy"] == pytest.approx(2 / 3)


def test_resolve_picks_named_metrics():
    picked = resolve(["accuracy", "f1"])
    assert set(picked) == {"accuracy", "f1"}


def test_resolve_names_the_ones_that_do_not_exist():
    with pytest.raises(ValueError, match="nope"):
        resolve(["accuracy", "nope"])


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
    """The sentinel replaces `float("-inf")` for both directions."""
    from core.metrics import Metric

    lower = Metric(name="rmse", fn=lambda y, yp: 0.0,
                   higher_is_better=False, bounds=(0.0, None))

    assert METRICS["accuracy"].better(0.0, METRICS["accuracy"].worst_sentinel)
    assert lower.better(1e9, lower.worst_sentinel)


def test_the_original_metrics_still_cost_exactly_one_minus_the_score():
    """Every `.ihpo` keeps its cost without a format bump."""
    from core.metrics import to_cost

    for name in LEGACY:
        for score in (0.0, 0.25, 0.5, 1.0):
            assert to_cost(METRICS[name], score) == 1.0 - score


def test_a_metric_that_is_already_a_cost_round_trips_untouched():
    """SMAC and the optimizer both minimise an imported objective."""
    from core.metrics import Metric, from_cost, to_cost

    cost = Metric(name="cost", fn=lambda y, yp: 0.0,
                  higher_is_better=False, bounds=(None, None))

    for value in (-3.5, 0.0, 42.7, 1e6):
        assert to_cost(cost, value) == value
        assert from_cost(cost, value) == value


def test_an_unbounded_higher_is_better_metric_is_negated():
    """Negation is exact, unlike a shift by an arbitrary constant."""
    from core.metrics import Metric, from_cost, to_cost

    m = Metric(name="gain", fn=lambda y, yp: 0.0, bounds=(0.0, None))

    assert to_cost(m, 3.5) == -3.5
    assert from_cost(m, -3.5) == 3.5


def test_cost_and_score_are_exact_inverses_for_every_shape():
    from core.metrics import Metric, from_cost, metric_for, to_cost

    shapes = [
        METRICS["accuracy"],
        Metric(name="rmse", fn=lambda *_: 0.0, higher_is_better=False, bounds=(0.0, None)),
        Metric(name="cost", fn=lambda *_: 0.0, higher_is_better=False, bounds=(None, None)),
        Metric(name="gain", fn=lambda *_: 0.0, bounds=(0.0, None)),
        metric_for("a name this build has never heard of"),
    ]
    for metric in shapes:
        for score in (-2.0, 0.0, 0.375, 17.0):
            assert from_cost(metric, to_cost(metric, score)) == pytest.approx(score)


def test_a_lower_is_better_cost_orders_the_way_the_optimizer_needs():
    """A better score must give a smaller cost, the only signal SMAC sees."""
    from core.metrics import Metric, to_cost

    for metric in (METRICS["accuracy"],
                   Metric(name="rmse", fn=lambda *_: 0.0,
                          higher_is_better=False, bounds=(0.0, None))):
        better, worse = (0.9, 0.1) if metric.higher_is_better else (0.1, 0.9)
        assert to_cost(metric, better) < to_cost(metric, worse)


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
    """`usable()` rules out the case on real runs, and the error marks a
    programming mistake."""
    from core.metrics import PROBABILITIES, Metric

    metrics = {"auc": Metric(name="auc", fn=lambda *_: 0.0, needs=PROBABILITIES)}

    with pytest.raises(ValueError, match="probabilities"):
        score_all(np.array(["a"]), ["a"], metrics)


def test_a_figure_titles_a_metric_by_its_label_or_its_capitalized_name():
    """Acronyms and R² carry a label; other metrics capitalize their name."""
    from core.metrics import undescribe

    assert [m.title for m in resolve(["rmse", "mae", "r2"]).values()] == ["RMSE", "MAE", "R²"]
    assert METRICS["accuracy"].title == "Accuracy"
    assert undescribe({"name": "logloss", "higher_is_better": False}).title == "Logloss"


def test_a_metrics_scale_runs_from_perfect_to_worthless():
    """A bounded metric's scale is its range, and an unbounded metric measures
    from its best end to the null score."""
    from core.metrics import METRICS as ALL, scale

    assert scale(ALL["accuracy"]) == 1.0
    assert scale(ALL["smape"]) == 200.0
    assert scale(ALL["rmse"], null_score=58.0) == 58.0
    assert scale(ALL["rmse"]) is None
    assert scale(ALL["rmse"], null_score=0.0) is None, "a null score at perfect measures nothing"
    assert scale(ALL["r2"], null_score=0.0) == 1.0
