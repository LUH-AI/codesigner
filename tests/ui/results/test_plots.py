"""The pure Plotly chart builders (ui/figures/figures.py).

One builder per figure that draws a figure, each named for its figure. These
assert the figures faithfully reflect a result — the incumbent line is the
running best, the scatter carries every trial's score, the selected point is
highlighted, the importance donut/bar mirror the importance dict, and the
duration bars are the per-trial times. Built from synthetic results with
hand-chosen numbers so every assertion is exact.
"""

import pytest

from core.optimizers import OptimizationResult, TrialResult
from ui.figures import (
    configuration_cube_plot,
    hyperparameter_ablation_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    incumbent_scores,
    parallel_coordinates_plot,
    partial_dependence_plot,
    performance_over_time_plot,
    trial_duration_plot,
)


def _result() -> OptimizationResult:
    """Three trials with accuracy 0.5, 0.3, 0.9 (so the running best is
    0.5, 0.5, 0.9) and importance defined for accuracy but not f1."""
    trials = [
        TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.5, "f1": 0.4},
                    score=0.5, incumbent_score=0.5, incumbent_config={"a": 1}),
        TrialResult(trial=2, config={"a": 2}, scores={"accuracy": 0.3, "f1": 0.2},
                    score=0.3, incumbent_score=0.5, incumbent_config={"a": 1}),
        TrialResult(trial=3, config={"a": 3}, scores={"accuracy": 0.9, "f1": 0.8},
                    score=0.9, incumbent_score=0.9, incumbent_config={"a": 3}),
    ]
    return OptimizationResult(
        trials=trials, primary_metric="accuracy",
        best_config={"a": 3}, best_score=0.9,
        hyperparameter_importance={"accuracy": {"a": 0.7, "b": 0.3}, "f1": {}},
        hyperparameter_importance_warning={"accuracy": None, "f1": "not enough trials"},
        trials_limit=None,
    )


def test_incumbent_scores_is_running_max():
    """incumbent_scores is the non-decreasing running best of the metric."""
    assert incumbent_scores(_result(), "accuracy") == [0.5, 0.5, 0.9]


def test_performance_figure_has_score_and_incumbent_traces():
    """The default view (trial x-axis, score y-axis) plots trial scores as
    markers and the incumbent as a line, both over the trial numbers.

    Expect: a markers trace and a lines trace among the figure's data; the
    marker trace's y equals the per-trial scores; the line trace's y equals
    the running best.
    """
    fig = performance_over_time_plot(_result(), "accuracy")
    markers = next(t for t in fig.data if t.mode == "markers" and t.name == "Trial score")
    line = next(t for t in fig.data if t.mode == "lines")
    assert list(markers.x) == [1, 2, 3]
    assert list(markers.y) == [0.5, 0.3, 0.9]
    assert list(line.y) == [0.5, 0.5, 0.9]


def test_performance_figure_highlights_selected_point():
    """The selected index is drawn larger and in the highlight color.

    Selecting trial index 2 must enlarge/recolor only that marker, leaving the
    others at the default size and color.
    """
    fig = performance_over_time_plot(_result(), "accuracy", selected_idx=2)
    markers = next(t for t in fig.data if t.mode == "markers" and t.name == "Trial score")
    assert markers.marker.size[2] > markers.marker.size[0]
    assert markers.marker.color[2] != markers.marker.color[0]


def test_performance_yaxis_labels_the_metric():
    """The y-axis is titled with the metric being shown, for the score view."""
    fig = performance_over_time_plot(_result(), "f1", y_axis="score")
    assert fig.layout.yaxis.title.text.lower() == "f1"


def test_time_axis_is_cumulative_trial_duration():
    """x_axis="time" sums each trial's duration rather than showing its index —
    compute spent, not wall-clock elapsed (which would include any gap between
    resumed runs)."""
    trials = [
        TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.5}, score=0.5,
                   incumbent_score=0.5, incumbent_config={"a": 1}, run_info={"time": 2.0}),
        TrialResult(trial=2, config={"a": 2}, scores={"accuracy": 0.9}, score=0.9,
                   incumbent_score=0.9, incumbent_config={"a": 2}, run_info={"time": 3.0}),
    ]
    result = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 2}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    fig = performance_over_time_plot(result, "accuracy", x_axis="time")
    markers = next(t for t in fig.data if t.mode == "markers")
    assert list(markers.x) == [2.0, 5.0]


def test_error_view_is_one_minus_score_on_a_log_axis():
    """y_axis="error" plots 1 - score (floored at 1e-3) with a log y-axis."""
    fig = performance_over_time_plot(_result(), "accuracy", y_axis="error")
    markers = next(t for t in fig.data if t.mode == "markers" and t.name == "Trial error")
    assert list(markers.y) == pytest.approx([0.5, 0.7, 0.1])
    assert fig.layout.yaxis.type == "log"


def test_performance_figure_none_when_no_trials():
    result = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert performance_over_time_plot(result, "accuracy") is None


def test_importance_pie_mirrors_the_importance_dict():
    """The default (pie) rendering's labels/values come straight from the
    importance dict handed in — any game's, since they're all the same shape."""
    fig = hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"])
    pie = fig.data[0]
    assert set(pie.labels) == {"a", "b"}
    assert dict(zip(pie.labels, pie.values)) == {"a": 0.7, "b": 0.3}


def test_importance_bar_mirrors_the_importance_dict():
    """The bar rendering carries the same numbers as pie, as vertical bars."""
    fig = hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"], "bar")
    bar = fig.data[0]
    assert dict(zip(bar.x, bar.y)) == {"a": 0.7, "b": 0.3}


def test_importance_table_rendering_draws_nothing():
    """The table rendering has no plot at all — the template renders it from
    `panels` directly, so the builder returns None for it same as "no data"."""
    assert hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"], "table") is None


def test_importance_figure_none_when_metric_has_no_importance():
    """hyperparameter_importance_plot returns None when handed an empty dict,
    so the view can show an explanatory message instead of an empty figure."""
    assert hyperparameter_importance_plot(_result().hyperparameter_importance["f1"]) is None


def test_figures_serialize_to_json():
    """Both figures survive fig.to_json() — the view embeds them that way."""
    assert performance_over_time_plot(_result(), "accuracy").to_json()
    assert hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"]).to_json()


def test_ablation_plot_is_signed_and_colored_by_sign():
    """Local ablation is a diverging bar: positive (helped vs. default) and
    negative (hurt) values both appear, colored differently, ranked by
    magnitude rather than sign."""
    fig = hyperparameter_ablation_plot({"a": -0.4, "b": 0.6, "c": 0.1})
    bar = fig.data[0]
    assert list(bar.x) == ["b", "a", "c"]  # |0.6| > |-0.4| > |0.1|
    assert list(bar.y) == [0.6, -0.4, 0.1]
    assert bar.marker.color[0] != bar.marker.color[1]  # positive vs negative


def test_ablation_plot_none_when_empty():
    assert hyperparameter_ablation_plot({}) is None


def _interactions():
    """Three hyperparameters: a strong synergy between a/b, a weaker one
    between a/c, and near-zero between b/c."""
    return {
        "a": {"a": 0.5, "b": 0.3, "c": 0.1},
        "b": {"a": 0.3, "b": 0.2, "c": -0.05},
        "c": {"a": 0.1, "b": -0.05, "c": 0.15},
    }


def test_interactions_heatmap_orders_by_diagonal_magnitude():
    """Hyperparameters are ordered by |diagonal value| descending: a (0.5),
    b (0.2), c (0.15) — and the grid mirrors the dict in that order."""
    fig = hyperparameter_interactions_heatmap_plot(_interactions())
    heatmap = fig.data[0]
    assert list(heatmap.x) == ["a", "b", "c"]
    assert list(heatmap.y) == ["a", "b", "c"]
    assert list(heatmap.z) == [[0.5, 0.3, 0.1], [0.3, 0.2, -0.05], [0.1, -0.05, 0.15]]


def test_interactions_heatmap_none_when_empty():
    assert hyperparameter_interactions_heatmap_plot({}) is None


def test_interactions_bar_ranks_off_diagonal_pairs_by_magnitude():
    """Only pairs (never the diagonal), ranked by |value| — a/b (0.3) beats
    a/c (0.1) beats b/c (-0.05)."""
    fig = hyperparameter_interactions_bar_plot(_interactions())
    bar = fig.data[0]
    assert list(bar.x) == ["a × b", "a × c", "b × c"]
    assert list(bar.y) == [0.3, 0.1, -0.05]
    assert bar.marker.color[0] == bar.marker.color[1]  # both positive
    assert bar.marker.color[2] != bar.marker.color[0]   # negative


def test_interactions_bar_respects_top_k():
    fig = hyperparameter_interactions_bar_plot(_interactions(), top_k=1)
    assert list(fig.data[0].x) == ["a × b"]


def test_interactions_bar_none_with_fewer_than_two_hyperparameters():
    assert hyperparameter_interactions_bar_plot({"a": {"a": 0.5}}) is None


def test_interactions_bar_none_when_empty():
    assert hyperparameter_interactions_bar_plot({}) is None


def _cube_result():
    """Three trials over two hyperparameters, with distinct values on every
    axis so a column mix-up would be caught."""
    trials = [
        TrialResult(trial=1, config={"a": 1, "b": 10}, scores={"accuracy": 0.5},
                    score=0.5, incumbent_score=0.5, incumbent_config={"a": 1, "b": 10}),
        TrialResult(trial=2, config={"a": 2, "b": 20}, scores={"accuracy": 0.3},
                    score=0.3, incumbent_score=0.5, incumbent_config={"a": 1, "b": 10}),
        TrialResult(trial=3, config={"a": 3, "b": 30}, scores={"accuracy": 0.9},
                    score=0.9, incumbent_score=0.9, incumbent_config={"a": 3, "b": 30}),
    ]
    return OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 3, "b": 30}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )


def test_configuration_cube_defaults_to_the_first_two_hyperparameters():
    """x/y come from the first two hyperparameters in config-space order, and
    every hyperparameter's values ride along in customdata for client-side
    axis remapping — see experiment_detail.html's applyCubeAxes."""
    fig = configuration_cube_plot(_cube_result(), "accuracy")
    trace = fig.data[0]
    assert list(trace.x) == [1, 2, 3]
    assert list(trace.y) == [10, 20, 30]
    assert [list(row) for row in trace.customdata] == [[1, 10], [2, 20], [3, 30]]
    assert fig.layout.meta["hp_names"] == ["a", "b"]


def test_configuration_cube_colors_by_the_given_metric():
    fig = configuration_cube_plot(_cube_result(), "accuracy")
    assert list(fig.data[0].marker.color) == [0.5, 0.3, 0.9]


def test_configuration_cube_none_with_no_trials():
    result = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert configuration_cube_plot(result, "accuracy") is None


def test_configuration_cube_none_with_fewer_than_two_hyperparameters():
    trials = [TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.5}, score=0.5,
                          incumbent_score=0.5, incumbent_config={"a": 1})]
    result = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 1}, best_score=0.5,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert configuration_cube_plot(result, "accuracy") is None


def _parcoords_result():
    """Three trials over a numeric hyperparameter (a), a boolean (b), and a
    string-categorical one (c) — enough to exercise both the numeric and the
    recoded-categorical dimension paths. Importance ranks b > a > c, so the
    ordered axes should read b, a, c, score."""
    trials = [
        TrialResult(trial=1, config={"a": 1, "b": True, "c": "x"}, scores={"accuracy": 0.5},
                    score=0.5, incumbent_score=0.5, incumbent_config={}),
        TrialResult(trial=2, config={"a": 2, "b": False, "c": "y"}, scores={"accuracy": 0.3},
                    score=0.3, incumbent_score=0.5, incumbent_config={}),
        TrialResult(trial=3, config={"a": 3, "b": True, "c": "x"}, scores={"accuracy": 0.9},
                    score=0.9, incumbent_score=0.9, incumbent_config={}),
    ]
    return OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 3, "b": True, "c": "x"},
        best_score=0.9,
        hyperparameter_importance={"accuracy": {"a": 0.3, "b": 0.6, "c": 0.1}},
        hyperparameter_importance_warning={"accuracy": None},
    )


def test_parallel_coordinates_orders_axes_by_importance():
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    labels = [d.label for d in fig.data[0].dimensions]
    assert labels == ["b", "a", "c", "Accuracy"]


def test_parallel_coordinates_keeps_numeric_axes_as_is():
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    dims = {d.label: d for d in fig.data[0].dimensions}
    assert list(dims["a"].values) == [1, 2, 3]
    assert "ticktext" not in dims["a"] or dims["a"].ticktext is None


def test_parallel_coordinates_recodes_non_numeric_axes():
    """Boolean and string hyperparameters become sorted-unique integer
    codes, with ticktext naming the original values — Parcoords dimensions
    are strictly numeric."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    dims = {d.label: d for d in fig.data[0].dimensions}

    b = dims["b"]
    assert list(b.ticktext) == ["False", "True"]  # sorted(..., key=str)
    assert list(b.values) == [1, 0, 1]  # True, False, True -> code 1, 0, 1

    c = dims["c"]
    assert list(c.ticktext) == ["x", "y"]
    assert list(c.values) == [0, 1, 0]


def test_parallel_coordinates_final_axis_is_the_score():
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    score_dim = fig.data[0].dimensions[-1]
    assert score_dim.label == "Accuracy"
    assert list(score_dim.values) == [0.5, 0.3, 0.9]
    assert list(fig.data[0].line.color) == [0.5, 0.3, 0.9]


def test_parallel_coordinates_none_with_no_trials():
    result = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert parallel_coordinates_plot(result, "accuracy") is None


def test_partial_dependence_plot_batches_ice_lines_into_one_none_separated_trace():
    """Every ICE row becomes one segment of a single trace, joined by a
    `None` (a gap, not a connecting line) — trace count must not grow with
    trial count."""
    fig = partial_dependence_plot(
        "max_depth", grid=[2, 5, 10], ice_lines=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], pdp=[0.25, 0.35, 0.45])
    ice_trace = fig.data[0]
    assert list(ice_trace.x) == [2, 5, 10, None, 2, 5, 10, None]
    assert list(ice_trace.y) == [0.1, 0.2, 0.3, None, 0.4, 0.5, 0.6, None]


def test_partial_dependence_plot_pdp_is_the_bold_second_trace():
    fig = partial_dependence_plot(
        "max_depth", grid=[2, 5, 10], ice_lines=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], pdp=[0.25, 0.35, 0.45])
    pdp_trace = fig.data[1]
    assert list(pdp_trace.x) == [2, 5, 10]
    assert list(pdp_trace.y) == [0.25, 0.35, 0.45]
    assert pdp_trace.line.width > fig.data[0].line.width


def test_partial_dependence_plot_none_with_empty_grid():
    assert partial_dependence_plot("max_depth", grid=[], ice_lines=[], pdp=[]) is None


def _timed(n, score, dur):
    return TrialResult(trial=n, config={"a": n}, scores={"accuracy": score}, score=score,
                       incumbent_score=score, incumbent_config={"a": n}, run_info={"time": dur})


def _timed_result(trials):
    return OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )


def test_trial_duration_figure_plots_per_trial_durations():
    """One bar per trial, its height the trial's measured duration."""
    fig = trial_duration_plot(_timed_result([_timed(1, 0.5, 0.2), _timed(2, 0.6, 0.3)]))
    trace = fig.to_dict()["data"][0]
    assert list(trace["x"]) == [1, 2]
    assert list(trace["y"]) == pytest.approx([0.2, 0.3])


def test_trial_duration_figure_none_when_no_trials():
    assert trial_duration_plot(_timed_result([])) is None
