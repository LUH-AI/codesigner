"""Timing figures: per-trial duration + error-vs-compute.

`duration_figure` is metric-independent (one bar per trial = its duration).
`error_vs_compute_figure` (remaining error vs cumulative compute time) is
per-metric and still being refined, so it's intentionally untested for now.
"""

import pytest

from core.optimizers.base import OptimizationResult, TrialResult
from ui.charts import duration_figure


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


# The three efficiency variants (error_reduction_spikes / regret_convergence /
# return_on_compute) are under visual comparison and intentionally untested for
# now — we'll add coverage once one is chosen.


def test_duration_figure_none_when_no_trials():
    assert duration_figure(_res([])) is None
