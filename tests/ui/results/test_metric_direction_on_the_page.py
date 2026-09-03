"""A metric where lower wins reads the right way round.

*What:* the display layer used to assume every score was a 0-to-1 number where
bigger is better — the error axis subtracted from 1.0, the absolute-scale toggle
pinned [0, 1], the colour ramps put dark at the top and parallel coordinates
drew the highest score last. None of that is true of an optimizer cost imported
from somebody else's run, which is lower-is-better and has no bounds at all.

*How:* a result is built declaring its own `smac:cost` metric — the mechanism an
imported run uses to say what its numbers mean — and each figure is asked what it
drew. The same assertions are made against `accuracy` to pin that nothing
changed for a metric that always worked.
"""

import pytest

from core.metrics import METRICS, Metric, error_range, to_error
from ui.figures.plots import (
    configuration_projection_plot, parallel_coordinates_plot,
    performance_over_time_plot,
)

#: Costs for four trials. The best is the smallest, and it is not the first —
#: so an incumbent line that steps the wrong way is visibly wrong.
COSTS = [0.8, 0.5, 0.9, 0.2]


def _result(metric_name, scores, *, declared=None):
    """An OptimizationResult over four trials scoring *scores* on *metric_name*."""
    from ui.registry import OPTIMIZERS

    optimizer = type(OPTIMIZERS["Random Search"])()
    return optimizer.deserialize_result({
        "data": [{"config_id": i + 1, "cost": s, "time": 0.1,
                  "scores": {metric_name: s}, "incumbent_config_id": i + 1}
                 for i, s in enumerate(scores)],
        "configs": {str(i + 1): {"depth": 2 + i, "rate": 0.1 * (i + 1)}
                    for i in range(len(scores))},
        "config_origins": {}, "primary_metric": metric_name,
        "declared_metrics": declared or {},
        "best_config_id": "1",
    })


def _a_cost_result():
    """Four trials scored by an imported, unbounded, lower-is-better objective."""
    return _result("smac:cost", COSTS, declared={
        "smac:cost": {"higher_is_better": False, "bounds": [None, None],
                      "needs": "labels"},
    })


def _an_accuracy_result():
    return _result("accuracy", [0.2, 0.5, 0.1, 0.8])


# ── what the metric itself says ─────────────────────────────────────────────

def test_a_cost_declared_by_a_file_reads_as_lower_is_better():
    """The declaration is what carries the direction — nothing infers it."""
    metric = _a_cost_result().metric("smac:cost")

    assert metric.higher_is_better is False
    assert metric.bounds == (None, None)
    assert metric.better(0.2, 0.5) is True


def test_a_cost_is_already_its_own_error():
    """`to_error` passes a lower-is-better score straight through.

    A metric where lower wins is a distance from perfect already; subtracting
    it from 1.0 would invent a different quantity and call it the same thing.
    """
    cost = Metric(name="c", fn=None, higher_is_better=False, bounds=(None, None))

    assert to_error(cost, 0.8) == 0.8
    assert to_error(METRICS["accuracy"], 0.8) == pytest.approx(0.2)


def test_an_unbounded_metric_has_no_error_range():
    """Nothing to pin an axis to, so the absolute toggle has nothing to offer."""
    unbounded = Metric(name="c", fn=None, higher_is_better=False,
                       bounds=(None, None))

    assert error_range(unbounded) is None
    assert error_range(METRICS["accuracy"]) == (0.0, 1.0)


# ── the performance figure ──────────────────────────────────────────────────

def test_the_incumbent_line_steps_down_for_a_cost():
    """The running best falls, and only at the two trials that improved on it."""
    fig = performance_over_time_plot(_a_cost_result(), "smac:cost")

    incumbent = next(t for t in fig.data if t.name == "Incumbent")
    assert list(incumbent.y) == [0.8, 0.5, 0.5, 0.2]

    improvements = next(t for t in fig.data if t.name == "New incumbent")
    assert list(improvements.x) == [1, 2, 4]


def test_the_incumbent_line_still_steps_up_for_an_accuracy():
    """The unchanged case, pinned so the fix cannot invert it."""
    fig = performance_over_time_plot(_an_accuracy_result(), "accuracy")

    incumbent = next(t for t in fig.data if t.name == "Incumbent")
    assert list(incumbent.y) == [0.2, 0.5, 0.5, 0.8]


def test_the_error_view_plots_a_cost_unchanged():
    """A cost is an error, so the error view draws the costs themselves."""
    fig = performance_over_time_plot(_a_cost_result(), "smac:cost", y_axis="error")

    assert list(fig.data[0].y) == COSTS


def test_the_error_view_still_inverts_an_accuracy():
    fig = performance_over_time_plot(_an_accuracy_result(), "accuracy",
                                     y_axis="error")

    assert list(fig.data[0].y) == pytest.approx([0.8, 0.5, 0.9, 0.2])


def test_the_error_view_declines_an_unbounded_higher_is_better_metric():
    """No fixed best to measure a distance from, so there is no figure.

    Declining is the honest answer: the alternative is an axis whose zero is
    wherever the data happened to stop, labelled "Error".
    """
    result = _result("r2", [0.2, 0.5, 0.1, 0.8], declared={
        "r2": {"higher_is_better": True, "bounds": [None, None],
               "needs": "labels"},
    })

    assert performance_over_time_plot(result, "r2", y_axis="error") is None
    assert performance_over_time_plot(result, "r2", y_axis="score") is not None


# ── the colour ramps ────────────────────────────────────────────────────────

def test_the_projection_reverses_its_ramp_for_a_cost():
    """Darker means better on every score-carrying figure, in either direction."""
    for metric, result, reversed_ in (
        ("smac:cost", _a_cost_result(), True),
        ("accuracy", _an_accuracy_result(), False),
    ):
        fig = configuration_projection_plot(result, metric, "pca")
        assert fig.data[0].marker.reversescale is reversed_, metric


def test_parallel_coordinates_draws_the_best_trial_last_for_a_cost():
    """Trace order is z-order, so the best line has to be added last.

    "Best" is the smallest cost here, which is the reverse of the sort that was
    correct when every metric was higher-is-better.
    """
    fig = parallel_coordinates_plot(_a_cost_result(), "smac:cost")

    drawn = [t.name for t in fig.data if (t.name or "").startswith("Trial ")]
    assert drawn[-1] == "Trial 4"   # cost 0.2, the best
    assert drawn[0] == "Trial 3"    # cost 0.9, the worst


def test_parallel_coordinates_still_draws_the_highest_accuracy_last():
    fig = parallel_coordinates_plot(_an_accuracy_result(), "accuracy")

    drawn = [t.name for t in fig.data if (t.name or "").startswith("Trial ")]
    assert drawn[-1] == "Trial 4"   # accuracy 0.8, the best
    assert drawn[0] == "Trial 3"    # accuracy 0.1, the worst


# ── the selected-configuration panel ────────────────────────────────────────

def test_the_delta_against_the_best_is_never_an_improvement():
    """Whichever direction the metric runs.

    The panel colours this delta green or red. Compared against the best trial
    it can only ever be red — but on a cost the difference is *positive*, so a
    template asking `delta > 0` would paint it green.
    """
    from ui.views import _selected_panel_data

    cost = _selected_panel_data(_a_cost_result(), "smac:cost", 0)
    assert cost["delta"] > 0
    assert cost["delta_better"] is False

    accuracy = _selected_panel_data(_an_accuracy_result(), "accuracy", 0)
    assert accuracy["delta"] < 0
    assert accuracy["delta_better"] is False


def test_the_best_trial_is_the_cheapest_one():
    """`best_index` follows the metric, so the panel points at trial 4."""
    from ui.views import _selected_panel_data

    assert _selected_panel_data(_a_cost_result(), "smac:cost", 3)["is_best"]
