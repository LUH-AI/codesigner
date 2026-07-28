"""Two timing figures: per-trial duration, and gain per unit time.

`duration_figure` is metric-independent (one bar per trial = its duration).
`gain_per_time_figure` is per-metric: the incumbent improvement contributed by
each trial divided by that trial's duration — the marginal value of the trial.
Both return None when there are no trials.
"""

import pytest

from core.optimizers.base import OptimizationResult, TrialResult
from web.charts import duration_figure, gain_per_time_figure


def _res(trials):
    return OptimizationResult(
        trials=trials, primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
    )


def _t(n, score, dur):
    return TrialResult(trial=n, config={"a": n}, scores={"accuracy": score}, score=score,
                       incumbent_score=score, incumbent_config={"a": n}, run_info={"time": dur})


def test_duration_figure_plots_per_trial_durations():
    fig = duration_figure(_res([_t(1, 0.5, 0.2), _t(2, 0.6, 0.3)]))
    trace = fig.to_dict()["data"][0]
    assert list(trace["x"]) == [1, 2]
    assert list(trace["y"]) == pytest.approx([0.2, 0.3])


def test_gain_line_drops_first_trial_and_declines_without_improvement():
    # best flat at 0.6; cumulative time 1→2→3. Line plots trials 2,3 (first
    # dropped) at 0.6/2=0.3 then 0.6/3=0.2 — descending utility.
    fig = gain_per_time_figure(_res([_t(1, 0.6, 1.0), _t(2, 0.6, 1.0), _t(3, 0.6, 1.0)]), "accuracy")
    line = fig.to_dict()["data"][0]
    assert list(line["x"]) == [2, 3]
    assert list(line["y"]) == pytest.approx([0.3, 0.2])
    assert line["y"][0] > line["y"][1]


def test_gain_marks_new_incumbents():
    # incumbent improves at trial 2 (0.5→0.8), not trial 3 → one marker, at x=2.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.8, 1.0), _t(3, 0.8, 1.0)]), "accuracy")
    data = fig.to_dict()["data"]
    assert len(data) == 2                       # line + new-incumbent markers
    markers = data[1]
    assert markers["mode"] == "markers"
    assert list(markers["x"]) == [2]


def test_gain_no_marker_trace_when_nothing_improves():
    # a declining line is still shown (utility eroding), but no win markers.
    fig = gain_per_time_figure(_res([_t(1, 1.0, 0.1), _t(2, 0.9, 0.5)]), "accuracy")
    assert len(fig.to_dict()["data"]) == 1


def test_gain_none_with_fewer_than_two_trials():
    assert gain_per_time_figure(_res([_t(1, 0.5, 1.0)]), "accuracy") is None


def test_figures_none_when_no_trials():
    assert duration_figure(_res([])) is None
    assert gain_per_time_figure(_res([]), "accuracy") is None
