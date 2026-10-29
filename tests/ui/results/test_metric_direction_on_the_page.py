"""The tests declare a `smac:cost` metric, lower-is-better and unbounded, and
check what each figure draws, with `accuracy` as the control.
"""

from typing import Any

import plotly.graph_objects as go
import pytest

from core.metrics import METRICS, Metric, unscoreable, error_range, to_error
from ui.figures.plots import (
    configuration_projection_plot, parallel_coordinates_plot,
    performance_over_time_plot,
)

#: Best cost last, exposing a wrong-way incumbent line.
COSTS = [0.8, 0.5, 0.9, 0.2]


def _drawn(fig: go.Figure | None) -> Any:
    assert fig is not None
    return fig


def _result(metric_name, scores, *, declared=None):
    """`_result` scores each trial on *metric_name*."""
    from core.registry import OPTIMIZERS

    optimizer = OPTIMIZERS["Random Search"]()
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
    """`_a_cost_result` imports an unbounded cost."""
    return _result("smac:cost", COSTS, declared={
        "smac:cost": {"higher_is_better": False, "bounds": [None, None],
                      "needs": "labels"},
    })


def _an_accuracy_result():
    return _result("accuracy", [0.2, 0.5, 0.1, 0.8])


# the metric itself

def test_a_cost_declared_by_a_file_reads_as_lower_is_better():
    """The declaration carries the direction."""
    metric = _a_cost_result().metric("smac:cost")

    assert metric.higher_is_better is False
    assert metric.bounds == (None, None)
    assert metric.better(0.2, 0.5) is True


def test_a_cost_is_already_its_own_error():
    """A lower-is-better score is already a distance from perfect.
    """
    cost = Metric(name="c", fn=unscoreable, higher_is_better=False, bounds=(None, None))

    assert to_error(cost, 0.8) == 0.8
    assert to_error(METRICS["accuracy"], 0.8) == pytest.approx(0.2)


def test_an_unbounded_metric_has_no_error_range():
    """No bounds leave the absolute toggle nothing to pin."""
    unbounded = Metric(name="c", fn=unscoreable, higher_is_better=False,
                       bounds=(None, None))

    assert error_range(unbounded) is None
    assert error_range(METRICS["accuracy"]) == (0.0, 1.0)


# the performance figure

def test_the_incumbent_line_steps_down_for_a_cost():
    """The running best only falls at improvements."""
    fig = _drawn(performance_over_time_plot(_a_cost_result(), "smac:cost"))

    incumbent = next(t for t in fig.data if t.name == "Incumbent")
    assert list(incumbent.y) == [0.8, 0.5, 0.5, 0.2]

    improvements = next(t for t in fig.data if t.name == "New incumbent")
    assert list(improvements.x) == [1, 2, 4]


def test_the_incumbent_line_still_steps_up_for_an_accuracy():
    fig = _drawn(performance_over_time_plot(_an_accuracy_result(), "accuracy"))

    incumbent = next(t for t in fig.data if t.name == "Incumbent")
    assert list(incumbent.y) == [0.2, 0.5, 0.5, 0.8]


def test_the_error_view_plots_a_cost_unchanged():
    fig = _drawn(performance_over_time_plot(_a_cost_result(), "smac:cost", y_axis="error"))

    assert list(fig.data[0].y) == COSTS


def test_the_error_view_still_inverts_an_accuracy():
    fig = _drawn(performance_over_time_plot(_an_accuracy_result(), "accuracy",
                                            y_axis="error"))

    assert list(fig.data[0].y) == pytest.approx([0.8, 0.5, 0.9, 0.2])


def test_the_error_view_declines_an_unbounded_higher_is_better_metric():
    """Without a fixed best, an "Error" axis would have an arbitrary zero.
    """
    result = _result("r2", [0.2, 0.5, 0.1, 0.8], declared={
        "r2": {"higher_is_better": True, "bounds": [None, None],
               "needs": "labels"},
    })

    assert performance_over_time_plot(result, "r2", y_axis="error") is None
    assert performance_over_time_plot(result, "r2", y_axis="score") is not None


# colour ramps

def test_the_projection_reverses_its_ramp_for_a_cost():
    """Darker means better in either direction."""
    for metric, result, reversed_ in (
        ("smac:cost", _a_cost_result(), True),
        ("accuracy", _an_accuracy_result(), False),
    ):
        fig = _drawn(configuration_projection_plot(result, metric, "pca"))
        assert fig.data[0].marker.reversescale is reversed_, metric


def test_parallel_coordinates_draws_the_best_trial_last_for_a_cost():
    """`parallel_coordinates_plot` draws the smallest cost last, since trace
    order is z-order.
    """
    fig = _drawn(parallel_coordinates_plot(_a_cost_result(), "smac:cost"))

    drawn = [t.name for t in fig.data if (t.name or "").startswith("Trial ")]
    assert drawn[-1] == "Trial 4"   # best (0.2)
    assert drawn[0] == "Trial 3"    # worst (0.9)


def test_parallel_coordinates_still_draws_the_highest_accuracy_last():
    fig = _drawn(parallel_coordinates_plot(_an_accuracy_result(), "accuracy"))

    drawn = [t.name for t in fig.data if (t.name or "").startswith("Trial ")]
    assert drawn[-1] == "Trial 4"   # best (0.8)
    assert drawn[0] == "Trial 3"    # worst (0.1)


# selected-configuration panel

def test_the_delta_against_the_best_is_never_an_improvement():
    """On a cost the delta to the best is positive, so `delta_better` must stay
    False.
    """
    from ui.views.dashboard import _selected_panel_data

    cost = _selected_panel_data(_a_cost_result(), "smac:cost", 0)
    assert cost["delta"] > 0
    assert cost["delta_better"] is False

    accuracy = _selected_panel_data(_an_accuracy_result(), "accuracy", 0)
    assert accuracy["delta"] < 0
    assert accuracy["delta_better"] is False


def test_the_best_trial_is_the_cheapest_one():
    """`best_index` follows the metric."""
    from ui.views.dashboard import _selected_panel_data

    assert _selected_panel_data(_a_cost_result(), "smac:cost", 3)["is_best"]
