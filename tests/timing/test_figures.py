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


def test_efficiency_falls_without_improvement():
    # best-so-far flat at 0.6, cumulative time grows 1→2→3 → 0.6, 0.3, 0.2.
    fig = gain_per_time_figure(_res([_t(1, 0.6, 1.0), _t(2, 0.6, 1.0), _t(3, 0.6, 1.0)]), "accuracy")
    y = list(fig.to_dict()["data"][0]["y"])
    assert y[0] > y[1] > y[2]
    assert y == pytest.approx([0.6, 0.3, 0.2])


def test_efficiency_spikes_on_improvement():
    # a big improvement in little added time lifts best ÷ elapsed above the prior point.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.9, 0.1)]), "accuracy")
    y = list(fig.to_dict()["data"][0]["y"])
    assert y[1] > y[0]                              # spike up: 0.9/1.1 > 0.5/1.0
    assert y == pytest.approx([0.5, 0.9 / 1.1])


def test_efficiency_guards_zero_elapsed():
    # a single zero-duration trial must not divide-by-zero.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 0.0)]), "accuracy")
    assert list(fig.to_dict()["data"][0]["y"]) == [0.0]


def test_figures_none_when_no_trials():
    assert duration_figure(_res([])) is None
    assert gain_per_time_figure(_res([]), "accuracy") is None
