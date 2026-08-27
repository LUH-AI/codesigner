"""A trial that produced no measurement, on the figures that show it.

`evaluate_trial` records a failure rather than raising: the trial scores 0.0 on
every metric, its status says it crashed or timed out, and the reason (with a
traceback where there is one) goes in `additional_info`. Every figure therefore
reads a failure as a trial that scored zero, and without a mark a reader cannot
tell the two apart — which matters most exactly where it is least visible, since
a cluster of crashes in one region of the space looks like a cluster of real
minima that the importance games and the projections will explain as if measured.

These cover which figures mark them and how. Two shapes of mark, for two shapes
of figure: a cross where a trial is a point, a tint where it is a fill. And one
figure leaves them out — parallel coordinates, where a line to a fabricated score
would cross every other line on the way to the floor of the axis.

The marks are per-point styling on the existing trace rather than a second
trace, so a failure stays selectable: its traceback is reached by clicking it.
"""

import pytest

from core.optimizers import OptimizationResult, TrialResult
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT
from ui.figures.plots import (
    FAILURE_FILL, FAILURE_SIZE, FAILURE_SYMBOL, NEGATIVE_COLOR, SELECTION_COLOR,
    configuration_cube_plot, configuration_projection_plot,
    parallel_coordinates_plot, performance_over_time_plot, trial_duration_plot,
)


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
    """Five trials, whichever of them *failed_at* names recorded as crashed.

    Enough trials for a projection (which needs three) and enough
    hyperparameters for one to have somewhere to project from.
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


# ── the trial itself ─────────────────────────────────────────────────────────

def test_a_trial_knows_whether_it_failed():
    """One property every figure and template asks, rather than each of them
    knowing which status ints mean what."""
    good, bad = _trial(1, 0.8), _trial(2, 0.0, status=STATUS_CRASHED, error="no")

    assert good.failed is False
    assert bad.failed is True
    assert bad.failure == "no"


def test_a_trial_with_no_status_counts_as_successful():
    """Every trial recorded before failures were tracked has none, and a run
    from an older .ihpo must not read as a run of failures."""
    assert _trial(1, 0.8).failed is False


def test_a_timeout_is_a_failure_like_any_other():
    """The figures do not distinguish them: both mean no measurement. Which it
    was is in the reason, for a reader who opens it."""
    timed_out = _trial(1, 0.0, status=STATUS_TIMEOUT, error="past its deadline")

    assert timed_out.failed is True
    assert timed_out.traceback == "", "nothing on this side is what went wrong"


# ── a cross, where a trial is a point ────────────────────────────────────────

@pytest.mark.parametrize("build", [
    pytest.param(lambda r: performance_over_time_plot(r, "accuracy"), id="performance"),
    pytest.param(lambda r: configuration_cube_plot(r, "accuracy"), id="cube"),
    pytest.param(lambda r: configuration_projection_plot(r, "accuracy", "pca"), id="projection"),
])
def test_a_failure_is_a_thick_red_cross(build):
    """`x-thin` has no fill by design — the glyph *is* the outline — so
    `marker.line` is what makes it visible and what makes it thick."""
    marker = build(_result(failed_at=(2,))).data[0].marker

    assert marker.symbol[1] == FAILURE_SYMBOL
    assert marker.symbol[0] != FAILURE_SYMBOL
    assert marker.line.color[1] == NEGATIVE_COLOR
    assert marker.line.width[1] == 3
    assert marker.size[1] >= FAILURE_SIZE


def test_a_run_with_no_failures_gains_no_per_point_arrays():
    """The overwhelmingly common case pays nothing: no symbol array, no line
    array, no extra bytes in the payload."""
    marker = performance_over_time_plot(_result(failed_at=()), "accuracy").data[0].marker

    assert marker.symbol is None


def test_the_cross_survives_being_selected_and_changes_colour():
    """Shape carries the failure, colour carries the selection. A reader who has
    clicked a failure still needs to see that it is the failure they clicked, so
    it keeps the cross and turns the selection colour."""
    marker = performance_over_time_plot(
        _result(failed_at=(2,)), "accuracy", selected_idx=1).data[0].marker

    assert marker.symbol[1] == FAILURE_SYMBOL, "still marked as a failure"
    assert marker.line.color[1] == SELECTION_COLOR, "and shown as selected"


def test_the_projection_keeps_its_score_scale_under_the_cross():
    """The fill stays on the metric scale — at 0.0, the worst end, which is what
    the trial scored. Recolouring it would cost the figure the value every other
    point is carrying."""
    marker = configuration_projection_plot(_result(), "accuracy", "pca").data[0].marker

    assert marker.colorscale is not None
    assert list(marker.color) == [0.55, 0.0, 0.65, 0.7, 0.75]


def test_a_failure_stays_selectable():
    """Which is why the mark is per-point styling and not a second trace. The
    traceback is reached by clicking the point, so the selection contract has to
    keep mapping every position to a trial."""
    figure = performance_over_time_plot(_result(failed_at=(2,)), "accuracy")
    positions = figure.layout.meta["selection"]["trials"]["0"]

    assert 1 in positions, "the failed trial is still one of the selectable points"
    assert len(positions) == 5


# ── a tint, where a trial is a fill ──────────────────────────────────────────

def test_a_failed_trial_keeps_its_duration_bar_and_is_tinted():
    """A crash is fast and a timeout is by definition the longest bar on the
    figure; either way the length is a real measurement, so the bar is tinted
    rather than crossed."""
    colors = list(trial_duration_plot(_result(failed_at=(2,))).data[0].marker.color)

    assert colors[1] == FAILURE_FILL
    assert colors[0] != FAILURE_FILL


# ── and one figure that leaves them out ──────────────────────────────────────

def test_parallel_coordinates_leaves_failures_out():
    """A line here claims the configuration scored what the last axis says. At a
    fabricated 0.0 it would run to the floor of that axis across every other
    line on the way."""
    figure = parallel_coordinates_plot(_result(failed_at=(2,)), "accuracy")
    drawn = figure.layout.meta["selection"]["trials"]

    assert len(drawn) == 4, "four of five trials"
    assert 1 not in [v[0] for v in drawn.values()]


def test_the_parallel_coordinates_highlight_still_points_at_a_real_trace():
    """It is indexed from what was drawn, not from what was run — so a run with
    failures must not leave the highlight writing into nothing."""
    figure = parallel_coordinates_plot(_result(failed_at=(2, 4)), "accuracy")

    assert figure.layout.meta["selection"]["highlight"] < len(figure.data)


def test_parallel_coordinates_scales_colour_to_the_trials_that_ran():
    """Anchored to a fabricated 0.0, every trial that did run would be squeezed
    into the top of the scale."""
    with_failure = parallel_coordinates_plot(_result(failed_at=(2,)), "accuracy")
    lines = [t.line.color for t in with_failure.data[:4]]

    assert len(set(lines)) == 4, "four distinct colours across the real range"


def test_parallel_coordinates_is_empty_when_every_trial_failed():
    """Nothing to draw and nothing to scale against."""
    assert parallel_coordinates_plot(
        _result(failed_at=(1, 2, 3, 4, 5)), "accuracy") is None
