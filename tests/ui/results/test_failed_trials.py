"""`evaluate_trial` records a failed trial as 0.0 on every metric, and every
figure would read that as a real zero. Point figures mark failures with a cross,
fill figures use a tint, and parallel coordinates drops them. A failure stays
clickable because each mark restyles one point of the existing trace.
"""

from typing import Any

import plotly.graph_objects as go
import pytest

from core.optimizers import OptimizationResult, TrialResult
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT
from ui.figures.plots import (
    FAILURE_FILL, FAILURE_LINE_WIDTH, FAILURE_LINE_WIDTH_RINGED, FAILURE_SIZE,
    FAILURE_SYMBOL, NEGATIVE_COLOR, SELECTION_COLOR,
    configuration_cube_plot, configuration_projection_plot,
    parallel_coordinates_plot, performance_over_time_plot, trial_duration_plot,
)


def _drawn(fig: go.Figure | None) -> Any:
    assert fig is not None
    return fig


def _trial(n, score, *, status=None, error="", traceback=""):
    info = {"time": 0.5 * n}
    if status is not None:
        info["status"] = status
        info["additional_info"] = {"error": error, "traceback": traceback}
    return TrialResult(
        trial=n, config={"a": n, "b": 10 - n},
        scores={"accuracy": score}, score=score,
        incumbent_score=score, incumbent_config={"a": n, "b": 10 - n},
        run_info=info)


def _result(failed_at=(2,)):
    """`_result` builds five trials and crashes each one in *failed_at*.
    """
    trials = []
    for n in range(1, 6):
        if n in failed_at:
            trials.append(_trial(n, 0.0, status=STATUS_CRASHED,
                                 error="ValueError: no", traceback="Traceback…"))
        else:
            trials.append(_trial(n, 0.5 + n / 20))
    return OptimizationResult(
        trials=trials, primary_metric="accuracy",
        best_config={"a": 5}, best_score=0.75,
        hyperparameter_importance={"accuracy": {"a": 0.6, "b": 0.4}},
        hyperparameter_importance_warning={"accuracy": None},
        trials_limit=None)


# the trial itself

def test_a_trial_knows_whether_it_failed():
    """`TrialResult.failed` spares figures and templates from decoding status
    ints."""
    good, bad = _trial(1, 0.8), _trial(2, 0.0, status=STATUS_CRASHED, error="no")

    assert good.failed is False
    assert bad.failed is True
    assert bad.failure == "no"


def test_a_trial_with_no_status_counts_as_successful():
    """Trials from older .ihpo files carry no status and must read as
    successful."""
    assert _trial(1, 0.8).failed is False


def test_a_timeout_is_a_failure_like_any_other():
    """The figures treat a timeout like a crash; the reason text says which."""
    timed_out = _trial(1, 0.0, status=STATUS_TIMEOUT, error="past its deadline")

    assert timed_out.failed is True
    assert timed_out.traceback == "", "nothing on this side is what went wrong"


# crosses on point figures

@pytest.mark.parametrize("build", [
    pytest.param(lambda r: performance_over_time_plot(r, "accuracy"), id="performance"),
    pytest.param(lambda r: configuration_cube_plot(r, "accuracy"), id="cube"),
    pytest.param(lambda r: configuration_projection_plot(r, "accuracy", "pca"), id="projection"),
])
def test_a_failure_is_a_red_cross_heavier_than_the_points_around_it(build):
    """`x-thin` has no fill, so `marker.line` draws the cross, and the test
    compares its width with the figure's ordinary points.
    """
    marker = build(_result(failed_at=(2,))).data[0].marker

    assert marker.symbol[1] == FAILURE_SYMBOL
    assert marker.symbol[0] != FAILURE_SYMBOL
    assert marker.line.color[1] == NEGATIVE_COLOR
    assert marker.line.width[1] > marker.line.width[0]
    assert marker.size[1] >= FAILURE_SIZE


def test_the_ringed_figures_draw_a_lighter_cross():
    """`FAILURE_LINE_WIDTH_RINGED` lightens the cross on the cube and the
    projection, which already ring every point.
    """
    failed = _result(failed_at=(2,))

    performance = _drawn(performance_over_time_plot(failed, "accuracy")).data[0].marker
    cube = _drawn(configuration_cube_plot(failed, "accuracy")).data[0].marker
    projection = _drawn(configuration_projection_plot(failed, "accuracy", "pca")).data[0].marker

    assert performance.line.width[1] == FAILURE_LINE_WIDTH
    assert cube.line.width[1] == FAILURE_LINE_WIDTH_RINGED
    assert projection.line.width[1] == FAILURE_LINE_WIDTH_RINGED
    assert FAILURE_LINE_WIDTH_RINGED < FAILURE_LINE_WIDTH


def test_a_run_with_no_failures_gains_no_per_point_arrays():
    marker = _drawn(performance_over_time_plot(_result(failed_at=()), "accuracy")).data[0].marker

    assert marker.symbol is None


def test_the_cross_survives_being_selected_and_changes_colour():
    """A selected failure keeps the cross shape and takes the selection colour."""
    marker = _drawn(performance_over_time_plot(
        _result(failed_at=(2,)), "accuracy", selected_idx=1)).data[0].marker

    assert marker.symbol[1] == FAILURE_SYMBOL, "still marked as a failure"
    assert marker.line.color[1] == SELECTION_COLOR, "and shown as selected"


def test_the_projection_keeps_its_score_scale_under_the_cross():
    """The projection fill stays on the metric scale at 0.0, the score the trial
    got."""
    marker = _drawn(configuration_projection_plot(_result(), "accuracy", "pca")).data[0].marker

    assert marker.colorscale is not None
    assert list(marker.color) == [0.55, 0.0, 0.65, 0.7, 0.75]


def test_a_failure_stays_selectable():
    """The selection map must cover a failed point too, because a click on it
    shows the traceback."""
    figure = _drawn(performance_over_time_plot(_result(failed_at=(2,)), "accuracy"))
    positions = figure.layout.meta["selection"]["trials"]["0"]

    assert 1 in positions, "the failed trial is still one of the selectable points"
    assert len(positions) == 5


# tints on fill figures

def test_a_failed_trial_keeps_its_duration_bar_and_is_tinted():
    """The bar is tinted instead of crossed because its length is a real
    duration."""
    colors = list(_drawn(trial_duration_plot(_result(failed_at=(2,)))).data[0].marker.color)

    assert colors[1] == FAILURE_FILL
    assert colors[0] != FAILURE_FILL


# figures that leave failures out

def test_parallel_coordinates_leaves_failures_out():
    """A line at a fabricated 0.0 would cross every other line on its way to the
    axis floor."""
    figure = _drawn(parallel_coordinates_plot(_result(failed_at=(2,)), "accuracy"))
    drawn = figure.layout.meta["selection"]["trials"]

    assert len(drawn) == 4, "four of five trials"
    assert 1 not in [v[0] for v in drawn.values()]


def test_the_parallel_coordinates_highlight_still_points_at_a_real_trace():
    """The highlight indexes the drawn traces and must still point at a real
    one."""
    figure = _drawn(parallel_coordinates_plot(_result(failed_at=(2, 4)), "accuracy"))

    assert figure.layout.meta["selection"]["highlight"] < len(figure.data)


def test_parallel_coordinates_scales_colour_to_the_trials_that_ran():
    """A 0.0 anchor would squeeze real trials into the scale's top."""
    with_failure = _drawn(parallel_coordinates_plot(_result(failed_at=(2,)), "accuracy"))
    lines = [t.line.color for t in with_failure.data[:4]]

    assert len(set(lines)) == 4, "four distinct colours across the real range"


def test_parallel_coordinates_is_empty_when_every_trial_failed():
    assert parallel_coordinates_plot(
        _result(failed_at=(1, 2, 3, 4, 5)), "accuracy") is None
