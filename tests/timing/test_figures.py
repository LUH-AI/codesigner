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


def test_gain_excludes_the_first_trial_and_spikes_at_improvements():
    # trials 1,2,3; the incumbent improves at trial 2 (0.5→0.8). Trial 1 is
    # dropped; trial 2 spikes (0.3/0.5 = 0.6), trial 3 (no improvement) is 0.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.8, 0.5), _t(3, 0.8, 1.0)]), "accuracy")
    trace = fig.to_dict()["data"][0]
    assert list(trace["x"]) == [2, 3]                 # first trial excluded
    assert list(trace["y"]) == pytest.approx([0.6, 0.0])


def test_gain_spike_height_is_improvement_over_duration():
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.9, 0.2)]), "accuracy")
    assert list(fig.to_dict()["data"][0]["y"]) == pytest.approx([2.0])  # (0.9-0.5)/0.2


def test_gain_uses_a_log_y_axis():
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.9, 0.2)]), "accuracy")
    assert fig.to_dict()["layout"]["yaxis"]["type"] == "log"


def test_gain_none_when_nothing_improves_after_first():
    # trial 1 is the best; no later trial beats it → nothing to show.
    assert gain_per_time_figure(_res([_t(1, 1.0, 0.1), _t(2, 0.9, 0.5)]), "accuracy") is None


def test_gain_none_with_fewer_than_two_trials():
    assert gain_per_time_figure(_res([_t(1, 0.5, 1.0)]), "accuracy") is None


def test_figures_none_when_no_trials():
    assert duration_figure(_res([])) is None
    assert gain_per_time_figure(_res([]), "accuracy") is None
