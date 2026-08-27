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


def test_ablation_plot_walks_from_the_default_to_this_trial():
    """A waterfall, not a diverging bar: the same signed values, but with the
    running total the reader actually wants — how far this trial got from the
    default configuration, and which changes carried it there.

    Ordered by magnitude so the steps that decided the outcome come first, and
    bracketed by labelled endpoints so the axis reads as "relative to the
    default" rather than as a score starting at zero.
    """
    fig = hyperparameter_ablation_plot({"a": -0.4, "b": 0.6, "c": 0.1})
    walk = fig.data[0]

    assert walk.type == "waterfall"
    assert list(walk.x) == ["Default", "b", "a", "c", "This trial"]  # |0.6|>|-0.4|>|0.1|
    assert list(walk.y) == [0.0, 0.6, -0.4, 0.1, 0.0]
    assert list(walk.measure) == ["absolute", "relative", "relative", "relative", "total"]


def test_ablation_plot_colours_gains_and_losses_apart():
    """A hyperparameter that hurt versus the default is a different finding
    from one that helped."""
    fig = hyperparameter_ablation_plot({"a": -0.4, "b": 0.6})

    assert (fig.data[0].increasing.marker.color
            != fig.data[0].decreasing.marker.color)


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


def _moebius():
    """Four terms over three hyperparameters, hand-chosen so every band is
    non-empty and the equal-split arithmetic is checkable by eye."""
    return [
        {"members": ["a"], "value": 0.5},
        {"members": ["b"], "value": 0.1},
        {"members": ["a", "b"], "value": -0.2},
        {"members": ["a", "b", "c"], "value": 0.3},
    ]


def test_graph_draws_a_line_per_pair_and_a_spoke_per_hyper_edge_member():
    """A pair is one line between two nodes; a coalition of three or more is a
    hub at their centroid with a spoke to each — the only way to show that they
    interact *as a set* rather than as three separate pairs."""
    fig = hyperparameter_graph_plot(_moebius())

    # a×b×c is three spokes; a×b is one line. Strongest first.
    assert len(fig.layout.shapes) == 3 + 1
    hubs = fig.data[0]
    assert list(hubs.text) == ["a × b × c<br>+0.3", "a × b<br>-0.2"]


def test_graph_colours_edges_by_sign_and_sizes_nodes_by_their_own_value():
    """Synergy and redundancy are different findings, and flattening the sign
    would lose the one the reader acts on."""
    fig = hyperparameter_graph_plot(_moebius())

    spoke, pair = fig.layout.shapes[0], fig.layout.shapes[-1]
    assert spoke.line.color != pair.line.color, "a×b×c is +, a×b is -"
    sizes = list(fig.data[1].marker.size)
    assert sizes[0] > sizes[1] > sizes[2], "a (0.5) > b (0.1) > c (nothing on its own)"


def test_graph_places_nodes_on_a_circle():
    """The whole layout — HyperSHAP pins its own to a circle too, which is what
    saves this from needing a force-directed algorithm Plotly does not have."""
    fig = hyperparameter_graph_plot(_moebius())
    nodes = fig.data[1]

    for x, y in zip(nodes.x, nodes.y):
        assert (x ** 2 + y ** 2) == pytest.approx(1.0)


def test_graph_caps_how_much_it_draws():
    """Six hyperparameters give 57 interactions and ten give 1,013; past a dozen
    the picture stops being a graph."""
    many = [{"members": ["a", f"h{i}"], "value": 1.0 / (i + 1)} for i in range(40)]
    fig = hyperparameter_graph_plot(many, top_k=5)

    assert len(fig.layout.shapes) == 5


def test_graph_none_when_nothing_can_interact():
    assert hyperparameter_graph_plot([]) is None
    assert hyperparameter_graph_plot([{"members": ["a"], "value": 1.0}]) is None


def test_upset_ranks_coalitions_and_keeps_their_sign():
    """Signed bars, so a redundant coalition hangs below the axis instead of
    being flattened to a magnitude."""
    fig = hyperparameter_upset_plot(_moebius())
    bar = fig.data[0]

    assert list(bar.x) == ["a × b × c", "a × b"], "strongest first"
    assert list(bar.y) == [0.3, -0.2], "signed, not flattened to a magnitude"


def test_upset_marks_exactly_the_members_of_each_coalition():
    fig = hyperparameter_upset_plot(_moebius())
    filled = set(zip(fig.data[-1].x, fig.data[-1].y))

    assert filled == {("a × b", "a"), ("a × b", "b"),
                      ("a × b × c", "a"), ("a × b × c", "b"), ("a × b × c", "c")}


def test_upset_none_without_a_coalition_to_show():
    """Order-1 terms are not coalitions — a run whose hyperparameters never
    interact has nothing for this view."""
    assert hyperparameter_upset_plot([{"members": ["a"], "value": 1.0}]) is None
    assert hyperparameter_upset_plot([]) is None


def test_orders_plot_splits_each_term_equally_among_its_members():
    """A Möbius term belongs to the coalition, not to any one member, so it is
    divided by the coalition's size — the standard reading of a Harsanyi
    dividend.

    a: 0.5 alone, 0.2/2 in the pair, 0.3/3 in the triple.
    c: nothing but 0.3/3.
    """
    bands = {t.name: dict(zip(t.x, t.y)) for t in hyperparameter_orders_plot(_moebius()).data}

    assert bands["On its own"] == pytest.approx({"a": 0.5, "b": 0.1, "c": 0.0})
    assert bands["In pairs"] == pytest.approx({"a": 0.1, "b": 0.1, "c": 0.0})
    assert bands["In larger groups"] == pytest.approx({"a": 0.1, "b": 0.1, "c": 0.1})


def test_orders_plot_uses_magnitude_so_signs_cannot_cancel():
    """The a/b pair is negative. Stacking it signed would subtract from a's bar
    and show a hyperparameter that matters in both directions as one that does
    not matter."""
    bands = {t.name: dict(zip(t.x, t.y)) for t in hyperparameter_orders_plot(_moebius()).data}

    assert bands["In pairs"]["a"] > 0


def test_orders_plot_stacks_and_ranks_by_total():
    fig = hyperparameter_orders_plot(_moebius())

    assert fig.layout.barmode == "stack"
    assert list(fig.data[0].x) == ["a", "b", "c"], "tallest total first"


def test_orders_plot_none_without_moebius_terms():
    """A run from before the decomposition was stored, or a game that fell back
    to the surrogate — both reach the figure's existing empty caption."""
    assert hyperparameter_orders_plot([]) is None


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


def test_configuration_cube_ships_no_axis_assignment_of_its_own():
    """Nothing is on an axis until someone puts it there — the three pickers
    start at None, so a server-chosen pair would answer the question the figure
    exists to ask. Every hyperparameter's values ride along in customdata, keyed
    by hp_names, for the client to fill the axes from (applyCubeAxes)."""
    fig = configuration_cube_plot(_cube_result(), "accuracy")
    trace = fig.data[0]
    assert list(trace.x) == []
    assert list(trace.y) == []
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


def test_configuration_cube_draws_for_a_single_hyperparameter():
    """One axis is a real answer to "where did the search go", and it is the
    client that decides how many axes to use anyway — so refusing here would
    leave a one-hyperparameter model with no cube at all rather than a strip."""
    trials = [TrialResult(trial=1, config={"a": 1}, scores={"accuracy": 0.5}, score=0.5,
                          incumbent_score=0.5, incumbent_config={"a": 1})]
    result = OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={"a": 1}, best_score=0.5,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )
    fig = configuration_cube_plot(result, "accuracy")

    assert fig.layout.meta["hp_names"] == ["a"]
    assert [list(row) for row in fig.data[0].customdata] == [[1]]


def _svm_result():
    """Three trials over the SVM model's own hyperparameters, two of which
    (`C`, `tol`) it searches logarithmically. Values span the full range so a
    linear axis would visibly bunch them."""
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
    """`C` is sampled log-uniformly over 0.01-100, so three quarters of a real
    run's points land in the bottom tenth of a linear axis — precisely the
    region the search spent its time in.

    The axes are assigned in the browser (applyCubeAxes), so this is the only
    place that fact can live: with it missing, whichever axis `C` lands on is
    silently linear.
    """
    fig = configuration_cube_plot(_svm_result(), "accuracy", config_space=_svm_space())

    assert fig.layout.meta["log_hps"] == ["C", "tol"]
    assert "kernel" not in fig.layout.meta["log_hps"]


def test_configuration_cube_without_a_config_space_stays_linear():
    """A custom model viewed read-only has nothing to ask, and a linear axis
    showing real values is a better answer than refusing to draw."""
    fig = configuration_cube_plot(_svm_result(), "accuracy", config_space=None)

    assert fig.layout.meta["log_hps"] == []


def test_configuration_cube_values_are_untouched_by_the_log_axis():
    """The axis is what changes, not the data — a log *axis* keeps the tick
    values real numbers, which is why this is preferred over encoding them.
    Read off customdata, which is where the values are now that the axes are
    filled in by the client."""
    fig = configuration_cube_plot(_svm_result(), "accuracy", config_space=_svm_space())

    assert [row[0] for row in fig.data[0].customdata] == [0.01, 1.0, 100.0]


def _axis_labels(fig):
    """The parallel-coordinates axes, in the order they are drawn."""
    return list(fig.layout.xaxis.ticktext)


def _axis_ticks(fig, label):
    """`(position, text)` for one axis's value labels, bottom to top.

    Every axis is drawn on one shared 0-1 scale with its own numbers written
    beside it, so the positions are where the values landed and the text is
    what they are called.
    """
    x = _axis_labels(fig).index(label)
    return sorted((a.y, a.text) for a in fig.layout.annotations if a.x == x)


def _trace_of_trial(fig):
    """Which trace draws which trial. They are not the same order: traces are
    added worst-scoring first, so the best lines draw on top."""
    sel = fig.layout.meta["selection"]
    return {trials[0]: int(trace) for trace, trials in sel["trials"].items()}


def _axis_positions(fig, label):
    """Where each trial sits on one axis, in trial order.

    A polyline carries several points along each span between two axes so that
    a click can land on the line and not only on an axis (see `_densified`), so
    the axis's own point is that many apart.
    """
    from ui.figures.plots import _PARCOORDS_STEPS

    x = _axis_labels(fig).index(label) * _PARCOORDS_STEPS
    traces = _trace_of_trial(fig)
    return [fig.data[traces[trial]].y[x] for trial in sorted(traces)]


def test_parallel_coordinates_recodes_a_log_axis_and_relabels_its_ticks():
    """Every axis shares one 0-1 scale, so a log hyperparameter goes on as
    log10 with the native numbers restored on the ticks — DeepCAVE's own
    approach, applied only where it is needed.

    Read off where the trials landed rather than off an encoded value: `C` is
    0.01, 1 and 100, which log-spaced are evenly apart and linearly would put
    the first two on top of each other at the bottom of the axis.
    """
    fig = parallel_coordinates_plot(_svm_result(), "accuracy", config_space=_svm_space())

    assert _axis_positions(fig, "C") == pytest.approx([0.0, 0.5, 1.0])
    assert [text for _y, text in _axis_ticks(fig, "C")] == \
        ["0.01", "0.1", "1", "10", "100"]


def test_parallel_coordinates_leaves_a_linear_axis_showing_its_own_values():
    """Only the log axes are recoded; everything else reads directly, as
    before — `a` is 1, 2, 3, so it lands evenly and its ticks say so."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy", config_space=None)

    assert _axis_positions(fig, "a") == pytest.approx([0.0, 0.5, 1.0])
    assert [text for _y, text in _axis_ticks(fig, "a")] == ["1", "1.5", "2", "2.5", "3"]


def test_parallel_coordinates_without_a_config_space_matches_the_old_output():
    """The fallback has to be exactly today's figure, not a near-miss: with
    nothing to ask about scales, `C`'s own values go on a linear axis and 0.01
    and 1 collapse into the bottom hundredth of it."""
    plain = parallel_coordinates_plot(_svm_result(), "accuracy", config_space=None)

    assert _axis_positions(plain, "C") == pytest.approx([0.0, 0.0099, 1.0], abs=1e-4)
    assert [text for _y, text in _axis_ticks(plain, "C")] == \
        ["0.01", "25.01", "50", "75", "100"]


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
    assert _axis_labels(fig) == ["b", "a", "c", "Accuracy"]


def test_parallel_coordinates_keeps_numeric_axes_as_is():
    """A numeric axis needs no recoding, so its ticks are its own values."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")

    assert [text for _y, text in _axis_ticks(fig, "a")] == ["1", "1.5", "2", "2.5", "3"]


def test_parallel_coordinates_recodes_non_numeric_axes():
    """Boolean and string hyperparameters become sorted-unique integer codes,
    with the tick naming the original value — a line has to have a height, and
    "True" is not one."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")

    assert [text for _y, text in _axis_ticks(fig, "b")] == ["False", "True"]
    # True, False, True -> code 1, 0, 1 -> top, bottom, top
    assert _axis_positions(fig, "b") == pytest.approx([1.0, 0.0, 1.0])

    assert [text for _y, text in _axis_ticks(fig, "c")] == ["x", "y"]
    assert _axis_positions(fig, "c") == pytest.approx([0.0, 1.0, 0.0])


def test_parallel_coordinates_final_axis_is_the_score():
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")

    assert _axis_labels(fig)[-1] == "Accuracy"
    # 0.5, 0.3, 0.9 over a 0.3-0.9 range
    assert _axis_positions(fig, "Accuracy") == pytest.approx([1 / 3, 0.0, 1.0])


def test_parallel_coordinates_colors_each_line_by_its_score():
    """The figure's whole reading is "a cluster of high-scoring lines bending
    through the same region", so the colour has to be per line. A Scatter line
    takes one colour rather than an array, which is why there is a trace per
    trial — see the builder."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    traces = _trace_of_trial(fig)
    colors = [fig.data[traces[trial]].line.color for trial in sorted(traces)]

    assert colors[1] == "#eef1fe", "trial 2 scores worst, palest"
    assert colors[2] == "#2b3aa8", "trial 3 scores best, deepest"
    assert len(set(colors)) == 3


def test_parallel_coordinates_draws_the_better_trials_on_top():
    """Trace order is z-order and nothing can change it after the fact, so with
    hundreds of lines crossing, the ones worth following have to be added last.
    In trial order they were whichever happened to run latest."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    traces = _trace_of_trial(fig)

    # scores are 0.5, 0.3, 0.9 for trials 0, 1, 2
    assert [traces[trial] for trial in (1, 0, 2)] == [0, 1, 2]


def test_parallel_coordinates_selection_names_one_trial_per_line():
    """A trial is a polyline here, not a point, so the click that names it
    lands on some vertex of the line and `segment` is what turns that back into
    the trial — see plots.py's `_selection_meta`."""
    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    sel = fig.layout.meta["selection"]

    assert sel["style"] == "line"
    # Four axes — three hyperparameters plus the score — with points along each
    # of the three spans between them, so a click lands on the line rather than
    # only within a few pixels of an axis.
    assert sel["segment"] == len(fig.data[0].x) == 13
    # One trial per trace, named — trace order is score order, not trial order.
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


# ── the selection contract ───────────────────────────────────────────────────
#
# One trial is selected at a time across the whole page, and every figure that
# draws all the trials both takes a click naming one and lights that one up.
# What makes that possible without the page knowing any figure by name is
# `layout.meta["selection"]` — see plots.py's `_selection_meta`.

def _selection_of(fig):
    return fig.layout.meta["selection"]


def test_every_trial_drawing_figure_says_where_its_trials_are():
    """The contract itself, over every builder that declares it: each named
    trace's list is as long as that trace has points, and every entry is a
    position that exists in the result. A figure that got this wrong would
    highlight the wrong trial rather than fail, which is why it is checked by
    construction rather than by eye."""
    result = _cube_result()
    n = len(result.trials)
    figures = [
        performance_over_time_plot(result, "accuracy"),
        configuration_cube_plot(result, "accuracy"),
        trial_duration_plot(result),
        parallel_coordinates_plot(result, "accuracy"),
    ]

    for fig in figures:
        sel = _selection_of(fig)
        assert sel["style"] in ("recolor", "outline", "line", "overlay")
        assert sel["trials"], "a figure that declares selection names its trials"
        for trace, trials in sel["trials"].items():
            data = fig.data[int(trace)]
            # The cube ships its values in customdata with the axes unassigned
            # (the pickers start at None), so that is where its point count is
            # until the client fills x and y in.
            points = len(data.x) or len(data.customdata or [])
            assert points == len(trials) * sel["segment"]
            assert all(0 <= i < n for i in trials)


def test_the_performance_figure_claims_only_its_outcome_trace():
    """Its other two traces are drawn from the same trials but are not a trial
    each — the incumbent is a running best and the diamonds are the subset that
    improved it, so a click on either names nothing."""
    sel = _selection_of(performance_over_time_plot(_cube_result(), "accuracy"))

    assert list(sel["trials"]) == ["0"]
    assert sel["trials"]["0"] == [0, 1, 2]


def test_neither_configuration_space_nor_the_beeswarm_recolours_a_point():
    """Both already spend colour on something — the score on one, the sign of
    the effect on the other — so painting the selected point a highlight colour
    would lose the value it was carrying.

    They answer that differently, because a ring is not always available: the
    beeswarm rings the point, and the projection draws the trial again on top,
    since Plotly draws no marker outline in a 3D scene and that is the view most
    in need of a landmark."""
    cube = _selection_of(configuration_cube_plot(_cube_result(), "accuracy"))
    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2}},
            {"index": 2, "trial": 3, "effects": {"a": -0.1}}]
    beeswarm = _selection_of(local_effects_plot(["a"], rows))

    assert cube["style"] == "overlay"
    assert cube["highlight"] == 1, "the trace the client adds for it"
    assert beeswarm["style"] == "outline"


def test_the_beeswarm_names_the_trials_it_sampled():
    """The one figure whose nth point is not its nth trial: it explains a
    sample, so position cannot imply the trial and the positions are carried.
    Every row is a point on every hyperparameter's own trace."""
    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2, "b": 0.1}},
            {"index": 2, "trial": 3, "effects": {"a": -0.1, "b": 0.0}}]
    sel = _selection_of(local_effects_plot(["a", "b"], rows))

    assert sorted(sel["trials"]) == ["0", "1"], "one trace per hyperparameter"
    assert sel["trials"]["0"] == [0, 2]
    assert sel["trials"]["1"] == [0, 2]


# ── the palette ──────────────────────────────────────────────────────────────
#
# Four colours, each meaning one thing, held by convention across every builder
# (see plots.py's palette block). A reader who learns a colour on one figure
# should not have to unlearn it on the next, so the places where two meanings
# would otherwise collide are pinned here.

def test_figures_scaled_to_the_metric_share_one_scale():
    """Both figures that colour by score use the same ramp, so "darker is
    better" is learned once. They are built by different code — one hands the
    array to Plotly, the other interpolates per line because a Scatter line
    takes a single colour — which is exactly how the two could drift apart."""
    from ui.figures.plots import _INTENSITY_SCALE, _scale_color

    result = _cube_result()
    cube = configuration_cube_plot(result, "accuracy")
    parcoords = parallel_coordinates_plot(result, "accuracy")
    colorbar = parcoords.data[len(result.trials)]

    assert cube.data[0].marker.colorscale == colorbar.marker.colorscale
    assert _scale_color(_INTENSITY_SCALE, 1.0) == _INTENSITY_SCALE[-1][1].lower()


def test_the_selection_color_says_only_that_a_trial_is_selected():
    """Three meanings meet on this one figure: a trial, the best-so-far, and
    the one that is selected. The incumbent line used to carry no colour of its
    own and took Plotly's second colorway entry, which is the same orange the
    selection is drawn in — so the highlight was competing with a line drawn
    through every point on the chart."""
    from ui.figures.plots import ACCENT_COLOR, MARKER_COLOR, SELECTION_COLOR

    fig = performance_over_time_plot(_cube_result(), "accuracy", selected_idx=1)

    assert len({SELECTION_COLOR, ACCENT_COLOR, MARKER_COLOR}) == 3
    # Translucent, and the same colour: the points are faded per point rather
    # than trace-wide, so a failed trial's cross can sit among them at full
    # strength. What matters here is that it is the selection colour and not
    # one of the other two.
    assert fig.data[0].marker.color[1] == _translucent(SELECTION_COLOR)
    assert fig.data[0].marker.color[0] == _translucent(MARKER_COLOR)
    assert fig.data[1].line.color == ACCENT_COLOR, "the incumbent line"
    assert fig.data[2].marker.color == ACCENT_COLOR, "the trials that improved it"


def test_the_negative_side_of_a_signed_figure_is_its_own_color():
    """Redundancy between hyperparameters and a value that hurt are the same
    reading, so they are the same colour — and neither is the selection or the
    best-so-far, which is what they would otherwise be confused with."""
    from ui.figures.plots import NEGATIVE_COLOR, SELECTION_COLOR

    rows = [{"index": 0, "trial": 1, "effects": {"a": 0.2, "b": -0.3}}]
    beeswarm = local_effects_plot(["a", "b"], rows)
    ablation = hyperparameter_ablation_plot({"a": 0.2, "b": -0.3})

    assert NEGATIVE_COLOR != SELECTION_COLOR
    assert NEGATIVE_COLOR in [t.marker.color[0] for t in beeswarm.data]
    assert ablation.data[0].decreasing.marker.color == NEGATIVE_COLOR


def test_the_beeswarm_rules_off_one_row_from_the_next():
    """Its points are jittered vertically so overlapping values stay countable,
    which leaves rows that run into each other exactly where one is widest —
    and a wide row is the finding this figure exists for."""
    rows = [{"index": i, "trial": i + 1, "effects": {"a": 0.1 * i, "b": -0.1 * i}}
            for i in range(3)]
    fig = local_effects_plot(["a", "b"], rows)

    horizontals = [s for s in fig.layout.shapes if s.y0 == s.y1]
    assert [s.y0 for s in horizontals] == [0.5], "one rule, between the two rows"


def test_a_parallel_coordinates_line_can_be_clicked_between_its_axes():
    """The reason the polylines carry more points than they have axes.

    With one point per axis a click only registers within a few pixels of an
    axis, and the whole middle of every span — most of the line — hits nothing.
    The extra points sit exactly on the straight segment, so the picture is
    unchanged and the click-to-trial rule still divides out to the same trial
    wherever on the line it lands.
    """
    from ui.figures.plots import _PARCOORDS_STEPS

    fig = parallel_coordinates_plot(_parcoords_result(), "accuracy")
    sel = fig.layout.meta["selection"]
    line = fig.data[1]

    assert len(line.x) == sel["segment"]
    for point in range(sel["segment"]):
        assert sel["trials"]["1"][point // sel["segment"]] == sel["trials"]["1"][0]

    # on the segment, not beside it: a quarter of the way along the first span
    at = _PARCOORDS_STEPS // 4
    assert line.y[at] == pytest.approx(
        line.y[0] + (line.y[_PARCOORDS_STEPS] - line.y[0]) * at / _PARCOORDS_STEPS)


def test_the_heatmap_reads_sign_the_same_way_the_graph_does():
    """Two views of one set of interactions, so blue has to mean the same thing
    on both.

    The heatmap used to ask for Plotly's named "RdBu", which the bundled
    plotly.min.js runs blue-to-red — putting positive on red, the reverse of the
    graph, the top-pairs bar, the waterfall and the beeswarm, with an orange
    band in the middle belonging to no meaning at all.
    """
    from ui.figures.plots import MARKER_COLOR, NEGATIVE_COLOR

    interactions = {"a": {"a": 0.4, "b": 0.2}, "b": {"a": 0.2, "b": 0.1}}
    heatmap = hyperparameter_interactions_heatmap_plot(interactions)
    moebius = [{"members": ["a"], "value": 0.4}, {"members": ["b"], "value": 0.1},
               {"members": ["a", "b"], "value": -0.2}]
    graph = hyperparameter_graph_plot(moebius)

    scale = heatmap.data[0].colorscale
    assert heatmap.data[0].zmid == 0
    assert scale[0][1].lower() == NEGATIVE_COLOR.lower(), "the low end is negative"
    assert scale[-1][1].lower() == MARKER_COLOR.lower(), "the high end is positive"
    # and the graph draws that same negative interaction in that same colour
    assert [s.line.color for s in graph.layout.shapes] == [NEGATIVE_COLOR]


def test_the_incumbent_markers_do_not_stand_between_the_reader_and_a_trial():
    """They are drawn over the trials they mark and are bigger than them, so
    with hover on they blocked clicks on exactly the trials most worth
    selecting — and being incumbents rather than trials, a click on one names
    nothing anyway. Out of the hit-testing, the point beneath answers."""
    fig = performance_over_time_plot(_cube_result(), "accuracy")

    assert fig.data[2].name == "New incumbent"
    assert fig.data[2].hoverinfo == "skip"
    assert fig.data[0].hoverinfo != "skip", "the trials themselves stay selectable"


def test_configuration_points_are_opaque_and_separated():
    """Two points that nearly overlap must not composite into something darker
    than either. On a scale where darker means better, that reads as a trial
    that did not happen — and the ring is what makes an overlap read as two
    points rather than as one good one."""
    for fig in (configuration_cube_plot(_cube_result(), "accuracy"),
                configuration_projection_plot(_cube_result(), "accuracy", "pca")):
        marker = fig.data[0].marker
        assert marker.opacity == 1
        assert marker.line.width == 1
        assert marker.line.color == "#FFFFFF"


def test_the_plain_importance_figure_is_untouched():
    """The split is its own builder, so the figure everyone sees before asking
    for anything is exactly what it was."""
    pie = hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "pie")
    bar = hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "bar")

    assert len(pie.data) == 1 and pie.data[0].type == "pie"
    assert len(bar.data) == 1 and bar.data[0].type == "bar"


def test_the_progress_bar_stacks_banked_under_what_is_left():
    """One quantity in two parts, so they stack rather than sit side by side —
    a bar that is nearly all light is a settled question, one that is nearly all
    dark is where the budget should go. Banked at the base, so it reads as a bar
    filling up rather than emptying out."""
    from ui.figures import hyperparameter_progress_plot

    rows = [{"name": "big", "achievable": 0.08, "banked": 0.071, "remaining": 0.009},
            {"name": "small", "achievable": 0.008, "banked": 0.001, "remaining": 0.007}]
    fig = hyperparameter_progress_plot(rows, "bar")

    assert [t.name for t in fig.data] == ["Already banked", "Still to gain"]
    assert list(fig.data[0].y) == pytest.approx([0.071, 0.001])
    assert list(fig.data[1].y) == pytest.approx([0.009, 0.007])
    assert fig.layout.barmode == "stack"
    # ordered by achievable, like the plain figure, so the two are comparable
    assert list(fig.data[0].x) == ["big", "small"]


def test_the_progress_pie_splits_each_wedge_in_place():
    """One flat pie with two adjacent slices per hyperparameter, so a wedge is
    literally part light and part dark.

    This was a sunburst first, which puts the split on an outer ring. Legible
    enough, but with several hyperparameters there is no single root node to
    fill the centre, so it leaves a hole there and puts the labels on the
    innermost ring where there is least room. A flat pie has neither problem and
    asks one less idea of the reader.
    """
    from ui.figures import hyperparameter_progress_plot
    from ui.figures.plots import _PROGRESS_SHADES

    rows = [{"name": "a", "achievable": 0.08, "banked": 0.06, "remaining": 0.02}]
    trace = hyperparameter_progress_plot(rows, "pie").data[0]

    assert trace.type == "pie"
    assert trace.hole == 0
    assert not trace.sort, "the two halves of a wedge have to stay together"
    assert list(trace.values) == pytest.approx([0.06, 0.02])
    assert list(trace.marker.colors) == [_PROGRESS_SHADES[1], _PROGRESS_SHADES[0]]


def test_a_hyperparameter_is_named_once_where_there_is_room_for_it():
    """Both slices labelled would say it twice, and the smaller one labelled
    would often be a label with nowhere to go."""
    from ui.figures import hyperparameter_progress_plot

    mostly_banked = hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.1, "banked": 0.09, "remaining": 0.01}], "pie")
    mostly_left = hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.1, "banked": 0.01, "remaining": 0.09}], "pie")

    assert list(mostly_banked.data[0].text) == ["a", ""]
    assert list(mostly_left.data[0].text) == ["", "a"]


def test_a_banked_value_outside_the_achievable_is_clamped_for_drawing():
    """The ablation can exceed the max game's estimate or go negative — two
    estimates of the same surface — and neither is a length. The table keeps the
    true signed value; the picture cannot."""
    from ui.figures import hyperparameter_progress_plot

    over = hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.05, "banked": 0.09, "remaining": 0.0}], "bar")
    under = hyperparameter_progress_plot(
        [{"name": "a", "achievable": 0.05, "banked": -0.02, "remaining": 0.07}], "bar")

    assert list(over.data[0].y) == [0.05] and list(over.data[1].y) == [0.0]
    assert list(under.data[0].y) == [0.0] and list(under.data[1].y) == [0.05]


def test_a_wedge_label_is_one_size_or_absent():
    """Plotly scales a label to fit its own slice, so a thin wedge gets tiny
    text and a fat one gets large text — emphasis that is not there, and
    unreadable at the small end.

    It takes both settings to stop it: `textfont` fixes the size, and
    `uniformtext` with mode "hide" decides what happens to a label that no
    longer fits at it — dropped rather than shrunk back, with the hover still
    carrying it. `uniformtext` alone only hides, which is why the first attempt
    at this changed nothing.
    """
    from ui.figures.plots import _TEXT_SIZE

    pie = hyperparameter_importance_plot({"a": 0.8, "b": 0.2}, "pie")

    assert pie.data[0].textfont.size == _TEXT_SIZE
    assert pie.layout.uniformtext.mode == "hide"
    assert pie.layout.uniformtext.minsize == _TEXT_SIZE, \
        "below it Plotly would be scaling again, which is the thing prevented"


def test_the_pie_says_it_has_no_hole_rather_than_leaving_it_out():
    """`Plotly.react` diffs against what is already drawn, so an attribute that
    is merely absent is a weaker instruction than one set to its default."""
    pie = hyperparameter_importance_plot({"a": 1.0}, "pie")

    assert pie.data[0].hole == 0
