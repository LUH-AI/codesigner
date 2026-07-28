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


def test_gain_per_time_is_improvement_over_duration():
    # trial 1 establishes 0.5 in 1.0 s → 0.5/s; trial 2 lifts 0.5 → 0.7 (Δ0.2)
    # in 0.5 s → 0.4/s.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 1.0), _t(2, 0.7, 0.5)]), "accuracy")
    y = list(fig.to_dict()["data"][0]["y"])
    assert y[0] == pytest.approx(0.5)       # first trial: establishes the incumbent
    assert y[1] == pytest.approx(0.4)


def test_gain_per_time_zero_for_non_improving_trial():
    fig = gain_per_time_figure(_res([_t(1, 0.5, 0.5), _t(2, 0.4, 0.5)]), "accuracy")
    assert list(fig.to_dict()["data"][0]["y"])[1] == 0.0


def test_gain_per_time_guards_zero_duration():
    # zero-duration trial must not divide-by-zero.
    fig = gain_per_time_figure(_res([_t(1, 0.5, 0.0), _t(2, 0.9, 0.0)]), "accuracy")
    assert all(v == 0.0 for v in fig.to_dict()["data"][0]["y"])


def test_figures_none_when_no_trials():
    assert duration_figure(_res([])) is None
    assert gain_per_time_figure(_res([]), "accuracy") is None
