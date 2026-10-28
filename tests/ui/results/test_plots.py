"""`ui.figures.plots` has one builder per figure, and the tests use synthetic
results with hand-chosen numbers so every assertion is exact.
"""

from typing import Any

import plotly.graph_objects as go
import pytest

from core.optimizers import OptimizationResult, TrialResult
from ui.figures.plots import ACCENT_COLOR, _translucent
from ui.figures import (
    configuration_cube_plot,
    configuration_projection_plot,
    hyperparameter_ablation_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    hyperparameter_graph_plot,
    hyperparameter_orders_plot,
    hyperparameter_upset_plot,
    incumbent_scores,
    local_effects_plot,
    parallel_coordinates_plot,
    partial_dependence_plot,
    performance_over_time_plot,
    trial_duration_plot,
)


def _drawn(fig: go.Figure | None) -> Any:
    assert fig is not None
    return fig


def _result() -> OptimizationResult:
    """`_result` holds accuracies 0.5, 0.3 and 0.9 and importance for accuracy
    only."""
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
    assert incumbent_scores(_result(), "accuracy") == [0.5, 0.5, 0.9]


def test_performance_figure_has_score_and_incumbent_traces():
    """The default view draws trial scores as markers and the running best as a
    line.
    """
    fig = _drawn(performance_over_time_plot(_result(), "accuracy"))
    markers = next(t for t in fig.data if t.mode == "markers" and t.name == "Trial score")
    line = next(t for t in fig.data if t.mode == "lines")
    assert list(markers.x) == [1, 2, 3]
    assert list(markers.y) == [0.5, 0.3, 0.9]
    assert list(line.y) == [0.5, 0.5, 0.9]


def test_performance_figure_highlights_selected_point():
    """`performance_over_time_plot` grows and recolours only the selected
    marker.
    """
    fig = _drawn(performance_over_time_plot(_result(), "accuracy", selected_idx=2))
    markers = next(t for t in fig.data if t.mode == "markers" and t.name == "Trial score")
    assert markers.marker.size[2] > markers.marker.size[0]
    assert markers.marker.color[2] != markers.marker.color[0]


def test_performance_yaxis_labels_the_metric():
    fig = _drawn(performance_over_time_plot(_result(), "f1", y_axis="score"))
    assert fig.layout.yaxis.title.text.lower() == "f1"


def test_time_axis_is_cumulative_trial_duration():
    """The time axis shows cumulative trial duration, which leaves out gaps
    between resumed runs."""
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
    fig = _drawn(performance_over_time_plot(result, "accuracy", x_axis="time"))
    markers = next(t for t in fig.data if t.mode == "markers")
    assert list(markers.x) == [2.0, 5.0]


def test_error_view_is_one_minus_score_on_a_log_axis():
    """Error is 1 - score, floored at 1e-3."""
    fig = _drawn(performance_over_time_plot(_result(), "accuracy", y_axis="error"))
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
    """Every game's importance dict has one shape."""
    fig = _drawn(hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"]))
    pie = fig.data[0]
    assert set(pie.labels) == {"a", "b"}
    assert dict(zip(pie.labels, pie.values)) == {"a": 0.7, "b": 0.3}


def test_importance_bar_mirrors_the_importance_dict():
    fig = _drawn(hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"], "bar"))
    bar = fig.data[0]
    assert dict(zip(bar.x, bar.y)) == {"a": 0.7, "b": 0.3}


def test_importance_table_rendering_draws_nothing():
    """The template renders the table from `panels`, so the builder returns
    None."""
    assert hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"], "table") is None


def test_importance_figure_none_when_metric_has_no_importance():
    """The None lets the view show an explanatory message."""
    assert hyperparameter_importance_plot(_result().hyperparameter_importance["f1"]) is None


def test_figures_serialize_to_json():
    """The view embeds figures via `to_json()`."""
    assert _drawn(performance_over_time_plot(_result(), "accuracy")).to_json()
    assert _drawn(hyperparameter_importance_plot(_result().hyperparameter_importance["accuracy"])).to_json()


def test_ablation_plot_walks_from_the_default_to_this_trial():
    """A waterfall shows the running total from the default, ordered by
    magnitude and bracketed by labelled endpoints.
    """
    fig = _drawn(hyperparameter_ablation_plot({"a": -0.4, "b": 0.6, "c": 0.1}))
    walk = fig.data[0]

    assert walk.type == "waterfall"
    assert list(walk.x) == ["Default", "b", "a", "c", "This trial"]
    assert list(walk.y) == [0.0, 0.6, -0.4, 0.1, 0.0]
    assert list(walk.measure) == ["absolute", "relative", "relative", "relative", "total"]


def test_ablation_plot_colours_gains_and_losses_apart():
    """Hurting versus the default is a different finding from helping."""
    fig = _drawn(hyperparameter_ablation_plot({"a": -0.4, "b": 0.6}))

    assert (fig.data[0].increasing.marker.color
            != fig.data[0].decreasing.marker.color)


def test_ablation_plot_none_when_empty():
    assert hyperparameter_ablation_plot({}) is None


def _interactions():
    """`_interactions` holds strong a/b synergy, weaker a/c and near-zero b/c
    synergy."""
    return {
        "a": {"a": 0.5, "b": 0.3, "c": 0.1},
        "b": {"a": 0.3, "b": 0.2, "c": -0.05},
        "c": {"a": 0.1, "b": -0.05, "c": 0.15},
    }


def test_interactions_heatmap_orders_by_diagonal_magnitude():
    """The heatmap orders a, b, c by descending diagonal magnitude."""
    fig = _drawn(hyperparameter_interactions_heatmap_plot(_interactions()))
    heatmap = fig.data[0]
    assert list(heatmap.x) == ["a", "b", "c"]
    assert list(heatmap.y) == ["a", "b", "c"]
    assert list(heatmap.z) == [[0.5, 0.3, 0.1], [0.3, 0.2, -0.05], [0.1, -0.05, 0.15]]


def test_interactions_heatmap_none_when_empty():
    assert hyperparameter_interactions_heatmap_plot({}) is None


def test_interactions_bar_ranks_off_diagonal_pairs_by_magnitude():
    """The bar shows only off-diagonal pairs, ranked by magnitude."""
    fig = _drawn(hyperparameter_interactions_bar_plot(_interactions()))
    bar = fig.data[0]
    assert list(bar.x) == ["a × b", "a × c", "b × c"]
    assert list(bar.y) == [0.3, 0.1, -0.05]
    assert bar.marker.color[0] == bar.marker.color[1]
    assert bar.marker.color[2] != bar.marker.color[0]


def test_interactions_bar_respects_top_k():
    fig = _drawn(hyperparameter_interactions_bar_plot(_interactions(), top_k=1))
    assert list(fig.data[0].x) == ["a × b"]


def test_interactions_bar_none_with_fewer_than_two_hyperparameters():
    assert hyperparameter_interactions_bar_plot({"a": {"a": 0.5}}) is None


def test_interactions_bar_none_when_empty():
    assert hyperparameter_interactions_bar_plot({}) is None


def _moebius():
    """Four terms, every band non-empty, splits checkable by eye."""
    return [
        {"members": ["a"], "value": 0.5},
        {"members": ["b"], "value": 0.1},
        {"members": ["a", "b"], "value": -0.2},
        {"members": ["a", "b", "c"], "value": 0.3},
    ]


def test_graph_draws_a_line_per_pair_and_a_spoke_per_hyper_edge_member():
    """A coalition of three or more draws a hub with a spoke to each member,
    showing that it interacts as a set."""
    fig = _drawn(hyperparameter_graph_plot(_moebius()))

    # 3 spokes (a×b×c) + 1 line (a×b)
    assert len(fig.layout.shapes) == 3 + 1
    hubs = fig.data[0]
    assert list(hubs.text) == ["a × b × c<br>+0.3", "a × b<br>-0.2"]


def test_graph_colours_edges_by_sign_and_sizes_nodes_by_their_own_value():
    """Synergy and redundancy are different findings."""
    fig = _drawn(hyperparameter_graph_plot(_moebius()))

    spoke, pair = fig.layout.shapes[0], fig.layout.shapes[-1]
    assert spoke.line.color != pair.line.color, "a×b×c is +, a×b is -"
    sizes = list(fig.data[1].marker.size)
    assert sizes[0] > sizes[1] > sizes[2], "a (0.5) > b (0.1) > c (nothing on its own)"


def test_graph_places_nodes_on_a_circle():
    """HyperSHAP also pins nodes to a circle, which avoids a force-directed
    layout Plotly lacks."""
    fig = _drawn(hyperparameter_graph_plot(_moebius()))
    nodes = fig.data[1]

    for x, y in zip(nodes.x, nodes.y):
        assert (x ** 2 + y ** 2) == pytest.approx(1.0)


def test_graph_caps_how_much_it_draws():
    """More than a dozen interactions ruin the picture from
    `hyperparameter_graph_plot`."""
    many = [{"members": ["a", f"h{i}"], "value": 1.0 / (i + 1)} for i in range(40)]
    fig = _drawn(hyperparameter_graph_plot(many, top_k=5))

    assert len(fig.layout.shapes) == 5


def test_graph_none_when_nothing_can_interact():
    assert hyperparameter_graph_plot([]) is None
    assert hyperparameter_graph_plot([{"members": ["a"], "value": 1.0}]) is None


def test_upset_ranks_coalitions_and_keeps_their_sign():
    """Signed bars let a redundant coalition hang below the axis."""
    fig = _drawn(hyperparameter_upset_plot(_moebius()))
    bar = fig.data[0]

    assert list(bar.x) == ["a × b × c", "a × b"], "strongest first"
    assert list(bar.y) == [0.3, -0.2], "signed, not flattened to a magnitude"


def test_upset_marks_exactly_the_members_of_each_coalition():
    fig = _drawn(hyperparameter_upset_plot(_moebius()))
    filled = set(zip(fig.data[-1].x, fig.data[-1].y))

    assert filled == {("a × b", "a"), ("a × b", "b"),
                      ("a × b × c", "a"), ("a × b × c", "b"), ("a × b × c", "c")}


def test_upset_none_without_a_coalition_to_show():
    """Order-1 terms are no coalitions."""
    assert hyperparameter_upset_plot([{"members": ["a"], "value": 1.0}]) is None
    assert hyperparameter_upset_plot([]) is None


def test_orders_plot_splits_each_term_equally_among_its_members():
    """Each Möbius term is split equally among its members, the standard
    Harsanyi dividend reading.
    """
    bands = {t.name: dict(zip(t.x, t.y)) for t in _drawn(hyperparameter_orders_plot(_moebius())).data}

    assert bands["On its own"] == pytest.approx({"a": 0.5, "b": 0.1, "c": 0.0})
    assert bands["In pairs"] == pytest.approx({"a": 0.1, "b": 0.1, "c": 0.0})
    assert bands["In larger groups"] == pytest.approx({"a": 0.1, "b": 0.1, "c": 0.1})


def test_orders_plot_uses_magnitude_so_signs_cannot_cancel():
    """Signed stacking would let a's negative pair cancel its solo bar."""
    bands = {t.name: dict(zip(t.x, t.y)) for t in _drawn(hyperparameter_orders_plot(_moebius())).data}

    assert bands["In pairs"]["a"] > 0


def test_orders_plot_stacks_and_ranks_by_total():
    fig = _drawn(hyperparameter_orders_plot(_moebius()))

    assert fig.layout.barmode == "stack"
    assert list(fig.data[0].x) == ["a", "b", "c"], "tallest total first"


def test_orders_plot_none_without_moebius_terms():
    assert hyperparameter_orders_plot([]) is None


def _cube_result():
    """`_cube_result` gives each axis distinct values to catch a column mix-up."""
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


def test_configuration_cube_ships_no_axis_assignment_of_its_own():
    """The server leaves every axis empty because the pickers start at None, and
    customdata holds every value."""
    fig = _drawn(configuration_cube_plot(_cube_result(), "accuracy"))
    trace = fig.data[0]
    assert list(trace.x) == []
    assert list(trace.y) == []
    assert [list(row) for row in trace.customdata] == [[1, 10], [2, 20], [3, 30]]
    assert fig.layout.meta["hp_names"] == ["a", "b"]


def test_configuration_cube_colors_by_the_given_metric():
    fig = _drawn(configuration_cube_plot(_cube_result(), "accuracy"))
    assert list(fig.data[0].marker.color) == [0.5, 0.3, 0.9]


def test_configuration_cube_none_with_no_trials():
    result = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert configuration_cube_plot(result, "accuracy") is None


def test_configuration_cube_draws_for_a_single_hyperparameter():
    """The client chooses the axis count, so a single hyperparameter still gets
    a strip."""
    trials = [TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.5}, score=0.5,
                          incumbent_score=0.5, incumbent_config={"a": 1})]
    result = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 1}, best_score=0.5,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    fig = _drawn(configuration_cube_plot(result, "accuracy"))

    assert fig.layout.meta["hp_names"] == ["a"]
    assert [list(row) for row in fig.data[0].customdata] == [[1]]


def _svm_result():
    """`_svm_result` holds SVM trials spanning log-searched `C` and `tol`."""
    def trial(n, c, tol, kernel, score):
        return TrialResult(trial=n, config={"C": c, "tol": tol, "kernel": kernel},
                           scores={"accuracy": score}, score=score,
                           incumbent_score=score, incumbent_config={})
    trials = [trial(1, 0.01, 1e-5, "rbf", 0.5),
              trial(2, 1.0, 1e-3, "linear", 0.3),
              trial(3, 100.0, 0.1, "rbf", 0.9)]
    return OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={}, best_score=0.9,
        hyperparameter_importance={}, hyperparameter_importance_warning={})


def _svm_space():
    from core.models import SVMModel
    return SVMModel().get_config_space(seed=0)


def test_configuration_cube_names_the_log_hyperparameters_for_the_client():
    """The browser assigns the axes, so `log_hps` must tell it that `C` is
    log-uniform over 0.01-100.
    """
    fig = _drawn(configuration_cube_plot(_svm_result(), "accuracy", config_space=_svm_space()))

    assert fig.layout.meta["log_hps"] == ["C", "tol"]
    assert "kernel" not in fig.layout.meta["log_hps"]


def test_configuration_cube_without_a_config_space_stays_linear():
    """Without a config space, the figure draws a linear axis of real values
    instead of refusing."""
    fig = _drawn(configuration_cube_plot(_svm_result(), "accuracy", config_space=None))

    assert fig.layout.meta["log_hps"] == []


def test_configuration_cube_values_are_untouched_by_the_log_axis():
    """A log axis keeps the values real, so customdata holds them unchanged."""
    fig = _drawn(configuration_cube_plot(_svm_result(), "accuracy", config_space=_svm_space()))

    assert [row[0] for row in fig.data[0].customdata] == [0.01, 1.0, 100.0]


def _axis_labels(fig):
    """`_axis_labels` lists axes in drawn order."""
    return list(fig.layout.xaxis.ticktext)


def _axis_ticks(fig, label):
    """`_axis_ticks` returns `(position, text)` for one axis's ticks, bottom to
    top.
    """
    x = _axis_labels(fig).index(label)
    return sorted((a.y, a.text) for a in fig.layout.annotations if a.x == x)


def _trace_of_trial(fig):
    """`_trace_of_trial` maps each trial to its trace, and worst-scoring traces
    come first."""
    sel = fig.layout.meta["selection"]
    return {trials[0]: int(trace) for trace, trials in sel["trials"].items()}


def _axis_positions(fig, label):
    """`_axis_positions` returns each trial's position on one axis, skipping
    `_densified` span points.
    """
    from ui.figures.plots import _PARCOORDS_STEPS

    x = _axis_labels(fig).index(label) * _PARCOORDS_STEPS
    traces = _trace_of_trial(fig)
    return [fig.data[traces[trial]].y[x] for trial in sorted(traces)]


def test_parallel_coordinates_recodes_a_log_axis_and_relabels_its_ticks():
    """A log hyperparameter goes on the shared 0-1 scale as log10, with native
    numbers on the ticks.
    """
    fig = _drawn(parallel_coordinates_plot(_svm_result(), "accuracy", config_space=_svm_space()))

    assert _axis_positions(fig, "C") == pytest.approx([0.0, 0.5, 1.0])
    assert [text for _y, text in _axis_ticks(fig, "C")] == \
        ["0.01", "0.1", "1", "10", "100"]


def test_parallel_coordinates_leaves_a_linear_axis_showing_its_own_values():
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy", config_space=None))

    assert _axis_positions(fig, "a") == pytest.approx([0.0, 0.5, 1.0])
    assert [text for _y, text in _axis_ticks(fig, "a")] == ["1", "1.5", "2", "2.5", "3"]


def test_parallel_coordinates_without_a_config_space_matches_the_old_output():
    """Without a config space, `C` stays linear and 0.01 and 1 collapse at the
    bottom."""
    plain = _drawn(parallel_coordinates_plot(_svm_result(), "accuracy", config_space=None))

    assert _axis_positions(plain, "C") == pytest.approx([0.0, 0.0099, 1.0], abs=1e-4)
    assert [text for _y, text in _axis_ticks(plain, "C")] == \
        ["0.01", "25.01", "50", "75", "100"]


def _parcoords_result():
    """`_parcoords_result` ranks importance b, a, c over numeric, boolean and
    string values."""
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
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))
    assert _axis_labels(fig) == ["b", "a", "c", "Accuracy"]


def test_parallel_coordinates_keeps_numeric_axes_as_is():
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))

    assert [text for _y, text in _axis_ticks(fig, "a")] == ["1", "1.5", "2", "2.5", "3"]


def test_parallel_coordinates_recodes_non_numeric_axes():
    """Non-numeric values become sorted integer codes named on the ticks,
    because a line needs a height."""
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))

    assert [text for _y, text in _axis_ticks(fig, "b")] == ["False", "True"]
    # codes 1, 0, 1
    assert _axis_positions(fig, "b") == pytest.approx([1.0, 0.0, 1.0])

    assert [text for _y, text in _axis_ticks(fig, "c")] == ["x", "y"]
    assert _axis_positions(fig, "c") == pytest.approx([0.0, 1.0, 0.0])


def test_parallel_coordinates_final_axis_is_the_score():
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))

    assert _axis_labels(fig)[-1] == "Accuracy"
    # range 0.3-0.9
    assert _axis_positions(fig, "Accuracy") == pytest.approx([1 / 3, 0.0, 1.0])


def test_parallel_coordinates_colors_each_line_by_its_score():
    """Each trial gets its own trace coloured by score, because a Scatter line
    takes one colour."""
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))
    traces = _trace_of_trial(fig)
    colors = [fig.data[traces[trial]].line.color for trial in sorted(traces)]

    assert colors[1] == "#eef1fe", "trial 2 scores worst, palest"
    assert colors[2] == "#2b3aa8", "trial 3 scores best, deepest"
    assert len(set(colors)) == 3


def test_parallel_coordinates_draws_the_better_trials_on_top():
    """Trace order is z-order, so the better trials are added last."""
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))
    traces = _trace_of_trial(fig)

    # scores: 0.5, 0.3, 0.9
    assert [traces[trial] for trial in (1, 0, 2)] == [0, 1, 2]


def test_parallel_coordinates_selection_names_one_trial_per_line():
    """`segment` maps a click on any polyline vertex back to the trial."""
    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))
    sel = fig.layout.meta["selection"]

    assert sel["style"] == "line"
    # Four axes plus span points, clickable between axes.
    assert sel["segment"] == len(fig.data[0].x) == 13
    # One trial per trace, in score order.
    assert sel["trials"] == {"0": [1], "1": [0], "2": [2]}
    assert fig.data[sel["highlight"]].line.width > fig.data[0].line.width
    assert sel["highlight"] == len(fig.data) - 1, "drawn last, so it draws on top"


def test_parallel_coordinates_none_with_no_trials():
    result = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    assert parallel_coordinates_plot(result, "accuracy") is None


def test_partial_dependence_plot_batches_ice_lines_into_one_none_separated_trace():
    """`None` gaps join the ICE rows, keeping one trace at any trial count."""
    fig = _drawn(partial_dependence_plot(
        "max_depth", grid=[2, 5, 10], ice_lines=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], pdp=[0.25, 0.35, 0.45]))
    ice_trace = fig.data[0]
    assert list(ice_trace.x) == [2, 5, 10, None, 2, 5, 10, None]
    assert list(ice_trace.y) == [0.1, 0.2, 0.3, None, 0.4, 0.5, 0.6, None]


def test_partial_dependence_plot_pdp_is_the_bold_second_trace():
    fig = _drawn(partial_dependence_plot(
        "max_depth", grid=[2, 5, 10], ice_lines=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], pdp=[0.25, 0.35, 0.45]))
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
    fig = _drawn(trial_duration_plot(_timed_result([_timed(1, 0.5, 0.2), _timed(2, 0.6, 0.3)])))
    trace = fig.to_dict()["data"][0]
    assert list(trace["x"]) == [1, 2]
    assert list(trace["y"]) == pytest.approx([0.2, 0.3])


def test_trial_duration_figure_none_when_no_trials():
    assert trial_duration_plot(_timed_result([])) is None


# Every trial-drawing figure declares `layout.meta["selection"]`, so the page
# selects trials without knowing figures by name.

def _selection_of(fig):
    return fig.layout.meta["selection"]


def test_every_trial_drawing_figure_says_where_its_trials_are():
    """Each named trace's list must match its point count and hold valid
    positions, since a mistake highlights the wrong trial without failing."""
    result = _cube_result()
    n = len(result.trials)
    figures = [
        _drawn(performance_over_time_plot(result, "accuracy")),
        _drawn(configuration_cube_plot(result, "accuracy")),
        _drawn(trial_duration_plot(result)),
        _drawn(parallel_coordinates_plot(result, "accuracy")),
    ]

    for fig in figures:
        sel = _selection_of(fig)
        assert sel["style"] in ("recolor", "outline", "line", "overlay")
        assert sel["trials"], "a figure that declares selection names its trials"
        for trace, trials in sel["trials"].items():
            data = fig.data[int(trace)]
            # The cube keeps its points in customdata until the client fills the
            # x and y axes.
            points = len(data.x) or len(data.customdata or [])
            assert points == len(trials) * sel["segment"]
            assert all(0 <= i < n for i in trials)


def test_the_performance_figure_claims_only_its_outcome_trace():
    """The incumbent line and the improvement diamonds are no trials, so a click
    on them names nothing."""
    sel = _selection_of(performance_over_time_plot(_cube_result(), "accuracy"))

    assert list(sel["trials"]) == ["0"]
    assert sel["trials"]["0"] == [0, 1, 2]


def test_neither_configuration_space_nor_the_beeswarm_recolours_a_point():
    """The cube and the beeswarm spend colour on data. The beeswarm rings the
    selected point, and the cube overlays it, since a 3D scene draws no marker
    outline."""
    cube = _selection_of(configuration_cube_plot(_cube_result(), "accuracy"))
    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2}},
            {"index": 2, "trial": 3, "effects": {"a": -0.1}}]
    beeswarm = _selection_of(local_effects_plot(["a"], rows))

    assert cube["style"] == "overlay"
    assert cube["highlight"] == 1, "the trace the client adds for it"
    assert beeswarm["style"] == "outline"


def test_the_beeswarm_names_the_trials_it_sampled():
    """The beeswarm explains a sample, so it carries each point's trial
    position."""
    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2, "b": 0.1}},
            {"index": 2, "trial": 3, "effects": {"a": -0.1, "b": 0.0}}]
    sel = _selection_of(local_effects_plot(["a", "b"], rows))

    assert sorted(sel["trials"]) == ["0", "1"], "one trace per hyperparameter"
    assert sel["trials"]["0"] == [0, 2]
    assert sel["trials"]["1"] == [0, 2]


# Four colours each keep one meaning across every builder; the tests pin where
# two meanings could collide.

def test_figures_scaled_to_the_metric_share_one_scale():
    """Both score-coloured figures must share one ramp, though different code
    builds each."""
    from ui.figures.plots import _INTENSITY_SCALE, _scale_color

    result = _cube_result()
    cube = _drawn(configuration_cube_plot(result, "accuracy"))
    parcoords = _drawn(parallel_coordinates_plot(result, "accuracy"))
    colorbar = parcoords.data[len(result.trials)]

    assert cube.data[0].marker.colorscale == colorbar.marker.colorscale
    assert _scale_color(_INTENSITY_SCALE, 1.0) == _INTENSITY_SCALE[-1][1].lower()


def test_the_selection_color_says_only_that_a_trial_is_selected():
    """Selection, incumbent and marker colours must differ on the performance
    figure."""
    from ui.figures.plots import ACCENT_COLOR, MARKER_COLOR, SELECTION_COLOR

    fig = _drawn(performance_over_time_plot(_cube_result(), "accuracy", selected_idx=1))

    assert len({SELECTION_COLOR, ACCENT_COLOR, MARKER_COLOR}) == 3
    # Fading is per point, so a failure's cross and the selected point both stay
    # at full strength.
    assert fig.data[0].marker.color[1] == SELECTION_COLOR
    assert fig.data[0].marker.color[0] == _translucent(MARKER_COLOR)
    assert fig.data[0].marker.color[0] == _translucent(MARKER_COLOR)
    assert fig.data[1].line.color == ACCENT_COLOR, "the incumbent line"
    assert fig.data[2].marker.color == ACCENT_COLOR, "the trials that improved it"


def test_the_negative_side_of_a_signed_figure_is_its_own_color():
    """`NEGATIVE_COLOR` marks both redundancy and a harmful value, and it
    differs from the selection and incumbent colours."""
    from ui.figures.plots import NEGATIVE_COLOR, SELECTION_COLOR

    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2, "b": -0.3}}]
    beeswarm = _drawn(local_effects_plot(["a", "b"], rows))
    ablation = _drawn(hyperparameter_ablation_plot({"a": 0.2, "b": -0.3}))

    assert NEGATIVE_COLOR != SELECTION_COLOR
    assert NEGATIVE_COLOR in [t.marker.color[0] for t in beeswarm.data]
    assert ablation.data[0].decreasing.marker.color == NEGATIVE_COLOR


def test_the_beeswarm_rules_off_one_row_from_the_next():
    """Vertical jitter makes rows run into each other where they are widest, so
    a rule separates them."""
    rows = [{"index": i, "trial": i + 1, "effects": {"a": 0.1 * i, "b": -0.1 * i}}
            for i in range(3)]
    fig = _drawn(local_effects_plot(["a", "b"], rows))

    horizontals = [s for s in fig.layout.shapes if s.y0 == s.y1]
    assert [s.y0 for s in horizontals] == [0.5], "one rule, between the two rows"


def test_a_parallel_coordinates_line_can_be_clicked_between_its_axes():
    """Extra points on each straight span make the whole line clickable without
    changing the picture.
    """
    from ui.figures.plots import _PARCOORDS_STEPS

    fig = _drawn(parallel_coordinates_plot(_parcoords_result(), "accuracy"))
    sel = fig.layout.meta["selection"]
    line = fig.data[1]

    assert len(line.x) == sel["segment"]
    for point in range(sel["segment"]):
        assert sel["trials"]["1"][point // sel["segment"]] == sel["trials"]["1"][0]

    # a quarter along the first span
    at = _PARCOORDS_STEPS // 4
    assert line.y[at] == pytest.approx(
        line.y[0] + (line.y[_PARCOORDS_STEPS] - line.y[0]) * at / _PARCOORDS_STEPS)


def test_the_heatmap_reads_sign_the_same_way_the_graph_does():
    """Blue must mean positive on both the heatmap and the graph.
    """
    from ui.figures.plots import MARKER_COLOR, NEGATIVE_COLOR

    interactions = {"a": {"a": 0.4, "b": 0.2}, "b": {"a": 0.2, "b": 0.1}}
    heatmap = _drawn(hyperparameter_interactions_heatmap_plot(interactions))
    moebius = [{"members": ["a"], "value": 0.4}, {"members": ["b"], "value": 0.1},
               {"members": ["a", "b"], "value": -0.2}]
    graph = _drawn(hyperparameter_graph_plot(moebius))

    scale = heatmap.data[0].colorscale
    assert heatmap.data[0].zmid == 0
    assert scale[0][1].lower() == NEGATIVE_COLOR.lower(), "the low end is negative"
    assert scale[-1][1].lower() == MARKER_COLOR.lower(), "the high end is positive"
    # same negative interaction, same colour on the graph
    assert [s.line.color for s in graph.layout.shapes] == [NEGATIVE_COLOR]


def test_the_incumbent_markers_do_not_stand_between_the_reader_and_a_trial():
    """The incumbent markers skip hover, so clicks reach the trial beneath."""
    fig = _drawn(performance_over_time_plot(_cube_result(), "accuracy"))

    assert fig.data[2].name == "New incumbent"
    assert fig.data[2].hoverinfo == "skip"
    assert fig.data[0].hoverinfo != "skip", "the trials themselves stay selectable"


def test_configuration_points_are_opaque_and_separated():
    """Overlapping points must stay opaque and ringed, since compositing would
    read as a better trial."""
    for fig in (_drawn(configuration_cube_plot(_cube_result(), "accuracy")),
                _drawn(configuration_projection_plot(_cube_result(), "accuracy", "pca"))):
        marker = fig.data[0].marker
        assert marker.opacity == 1
        assert marker.line.width == 1
        assert marker.line.color == "#FFFFFF"


def test_the_plain_importance_figure_is_untouched():
    """The progress split has its own builder."""
    pie = _drawn(hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "pie"))
    bar = _drawn(hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "bar"))

    assert len(pie.data) == 1 and pie.data[0].type == "pie"
    assert len(bar.data) == 1 and bar.data[0].type == "bar"


def test_the_progress_bar_stacks_banked_under_what_is_left():
    """The figure stacks the banked part under the remaining part, so each bar
    fills up from its base."""
    from ui.figures import hyperparameter_progress_plot

    rows = [{"name": "big", "achievable": 0.08, "banked": 0.071, "remaining": 0.009},
            {"name": "small", "achievable": 0.008, "banked": 0.001, "remaining": 0.007}]
    fig = _drawn(hyperparameter_progress_plot(rows, "bar"))

    assert [t.name for t in fig.data] == ["Already banked", "Still to gain"]
    assert list(fig.data[0].y) == pytest.approx([0.071, 0.001])
    assert list(fig.data[1].y) == pytest.approx([0.009, 0.007])
    assert fig.layout.barmode == "stack"
    # ordered by achievable, like the plain figure
    assert list(fig.data[0].x) == ["big", "small"]


def test_the_progress_pie_splits_each_wedge_in_place():
    """A flat pie gives each hyperparameter two adjacent slices, splitting the
    wedge in place.
    """
    from ui.figures import hyperparameter_progress_plot
    from ui.figures.plots import _PROGRESS_SHADES

    rows = [{"name": "a", "achievable": 0.08, "banked": 0.06, "remaining": 0.02}]
    trace = _drawn(hyperparameter_progress_plot(rows, "pie")).data[0]

    assert trace.type == "pie"
    assert trace.hole == 0
    assert not trace.sort, "the two halves of a wedge have to stay together"
    assert list(trace.values) == pytest.approx([0.06, 0.02])
    assert list(trace.marker.colors) == [_PROGRESS_SHADES[1], _PROGRESS_SHADES[0]]


def test_a_hyperparameter_is_named_once_where_there_is_room_for_it():
    """Only the larger slice carries the name."""
    from ui.figures import hyperparameter_progress_plot

    mostly_banked = _drawn(hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.1, "banked": 0.09, "remaining": 0.01}], "pie"))
    mostly_left = _drawn(hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.1, "banked": 0.01, "remaining": 0.09}], "pie"))

    assert list(mostly_banked.data[0].text) == ["a", ""]
    assert list(mostly_left.data[0].text) == ["", "a"]


def test_a_banked_value_outside_the_achievable_is_clamped_for_drawing():
    """The table keeps the signed value, and the bar is clamped to the range
    from zero to the max estimate."""
    from ui.figures import hyperparameter_progress_plot

    over = _drawn(hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.05, "banked": 0.09, "remaining": 0.0}], "bar"))
    under = _drawn(hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.05, "banked": -0.02, "remaining": 0.07}], "bar"))

    assert list(over.data[0].y) == [0.05] and list(over.data[1].y) == [0.0]
    assert list(under.data[0].y) == [0.0] and list(under.data[1].y) == [0.05]


def test_a_wedge_label_is_one_size_or_absent():
    """`textfont` fixes the label size, and `uniformtext` mode "hide" drops
    labels that no longer fit.
    """
    from ui.figures.plots import _TEXT_SIZE

    pie = _drawn(hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "pie"))

    assert pie.data[0].textfont.size == _TEXT_SIZE
    assert pie.layout.uniformtext.mode == "hide"
    assert pie.layout.uniformtext.minsize == _TEXT_SIZE, \
        "below it Plotly would be scaling again, which is the thing prevented"


def test_the_pie_says_it_has_no_hole_rather_than_leaving_it_out():
    """An explicit default beats an absent attribute, because `Plotly.react`
    diffs against the drawn figure."""
    pie = _drawn(hyperparameter_importance_plot({"a": 1.0}, "pie"))

    assert pie.data[0].hole == 0
