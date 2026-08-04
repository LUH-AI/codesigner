"""The pure Plotly figure builders (ui/charts/figures.py).

One builder per chart that draws a figure, each named for its chart. These
assert the figures faithfully reflect a result — the incumbent line is the
running best, the scatter carries every trial's score, the selected point is
highlighted, the importance donut mirrors the importance dict, and the duration
bars are the per-trial times. Built from synthetic results with hand-chosen
numbers so every assertion is exact.

`error_over_time_figure` is still being refined visually, so it is deliberately
uncovered for now.
"""

import pytest

from core.optimizers import OptimizationResult, TrialResult
from ui.charts import (
    hyperparameter_importance_figure,
    incumbent_performance_figure,
    incumbent_scores,
    trial_duration_figure,
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
    """The performance figure plots trial scores as markers and the incumbent
    as a line, both over the trial numbers.

    Expect: two traces; the marker trace's y equals the per-trial scores; the
    line trace's y equals the running best.
    """
    fig = incumbent_performance_figure(_result(), "accuracy")
    assert len(fig.data) == 2

    markers = next(t for t in fig.data if t.mode == "markers")
    line = next(t for t in fig.data if t.mode == "lines")
    assert list(markers.x) == [1, 2, 3]
    assert list(markers.y) == [0.5, 0.3, 0.9]
    assert list(line.y) == [0.5, 0.5, 0.9]


def test_performance_figure_highlights_selected_point():
    """The selected index is drawn larger and in the highlight color.

    Selecting trial index 2 must enlarge/recolor only that marker, leaving the
    others at the default size and color.
    """
    fig = incumbent_performance_figure(_result(), "accuracy", selected_idx=2)
    markers = next(t for t in fig.data if t.mode == "markers")
    assert markers.marker.size[2] > markers.marker.size[0]
    assert markers.marker.color[2] != markers.marker.color[0]


def test_performance_yaxis_labels_the_metric():
    """The y-axis is titled with the metric being shown."""
    fig = incumbent_performance_figure(_result(), "f1")
    assert fig.layout.yaxis.title.text.lower() == "f1"


def test_importance_figure_mirrors_the_importance_dict():
    """The importance donut's labels and values come straight from the result.

    A pie trace whose labels/values equal the hyperparameter_importance entry
    for the metric.
    """
    fig = hyperparameter_importance_figure(_result(), "accuracy")
    pie = fig.data[0]
    assert set(pie.labels) == {"a", "b"}
    assert dict(zip(pie.labels, pie.values)) == {"a": 0.7, "b": 0.3}


def test_importance_figure_none_when_metric_has_no_importance():
    """hyperparameter_importance_figure returns None when the metric has no importance data,
    so the view can show an explanatory message instead of an empty chart."""
    assert hyperparameter_importance_figure(_result(), "f1") is None


def test_figures_serialize_to_json():
    """Both figures survive fig.to_json() — the view embeds them that way."""
    assert incumbent_performance_figure(_result(), "accuracy").to_json()
    assert hyperparameter_importance_figure(_result(), "accuracy").to_json()


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
    fig = trial_duration_figure(_timed_result([_timed(1, 0.5, 0.2), _timed(2, 0.6, 0.3)]))
    trace = fig.to_dict()["data"][0]
    assert list(trace["x"]) == [1, 2]
    assert list(trace["y"]) == pytest.approx([0.2, 0.3])


def test_trial_duration_figure_none_when_no_trials():
    assert trial_duration_figure(_timed_result([])) is None
