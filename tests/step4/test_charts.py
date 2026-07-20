"""Step 4: the pure Plotly figure builders (web/charts.py).

These assert the figures faithfully reflect a result — the incumbent line is
the running best, the scatter carries every trial's score, the selected point
is highlighted, and the importance donut mirrors the importance dict. Built
from a synthetic result with hand-chosen numbers so every assertion is exact.
"""

import pytest

from core.optimizers import OptimizationResult, TrialResult
from web.charts import importance_figure, incumbent_scores, performance_figure


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
    fig = performance_figure(_result(), "accuracy")
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
    fig = performance_figure(_result(), "accuracy", selected_idx=2)
    markers = next(t for t in fig.data if t.mode == "markers")
    assert markers.marker.size[2] > markers.marker.size[0]
    assert markers.marker.color[2] != markers.marker.color[0]


def test_performance_yaxis_labels_the_metric():
    """The y-axis is titled with the metric being shown."""
    fig = performance_figure(_result(), "f1")
    assert fig.layout.yaxis.title.text.lower() == "f1"


def test_importance_figure_mirrors_the_importance_dict():
    """The importance donut's labels and values come straight from the result.

    A pie trace whose labels/values equal the hyperparameter_importance entry
    for the metric.
    """
    fig = importance_figure(_result(), "accuracy")
    pie = fig.data[0]
    assert set(pie.labels) == {"a", "b"}
    assert dict(zip(pie.labels, pie.values)) == {"a": 0.7, "b": 0.3}


def test_importance_figure_none_when_metric_has_no_importance():
    """importance_figure returns None when the metric has no importance data,
    so the view can show an explanatory message instead of an empty chart."""
    assert importance_figure(_result(), "f1") is None


def test_figures_serialize_to_json():
    """Both figures survive fig.to_json() — the view embeds them that way."""
    assert performance_figure(_result(), "accuracy").to_json()
    assert importance_figure(_result(), "accuracy").to_json()
