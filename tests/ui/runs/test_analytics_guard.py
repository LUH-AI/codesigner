"""When the at-run-completion importance analytics are skipped, and how they say so.

Two ways they get skipped, both landing in the per-metric
`hyperparameter_*_warning` fields the experiment page already renders:

- the run was cancelled — you should not wait minutes for analytics on a run you
  just stopped;
- the run is too wide to afford, against `ANALYTICS_EAGER_MAX_COALITIONS`.

The cost being guarded is exponential in *hyperparameter count* and independent
of trial count (a shapiq ExactComputer evaluates 2^n_hp coalitions per game), so
these use tiny trial counts and vary the config space instead — a short run of a
wide model is precisely the expensive case.
"""

import threading

import pytest
from ConfigSpace import ConfigurationSpace, Float

from core.models import RandomForestModel
from core.optimizers import RandomOptimizer
from core.optimizers.base import EAGER_MAX_COALITIONS


def _wide_space(n_hp, seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add([Float(f"h{i}", (0.0, 1.0), default=0.5) for i in range(n_hp)])
    return cs


# ── The budget arithmetic, without running anything expensive ────────────────

@pytest.mark.parametrize("n_hp,n_metrics,expected", [
    (4, 4, 192),      # Random Forest, all metrics — 3.4s measured
    (6, 4, 768),      # SVM Classifier — 13s measured
    (8, 4, 3072),     # ~51s
    (10, 4, 12288),   # ~3.5 minutes — the custom-upload pathology
])
def test_the_budget_counts_coalitions_across_games_and_metrics(n_hp, n_metrics, expected):
    """2^hyperparameters x 3 games x metrics. Pinned because the default is
    chosen against these numbers, so a change to either should be deliberate."""
    opt = RandomOptimizer()
    opt.analytics_max_coalitions = expected          # exactly affordable
    metrics = [f"m{i}" for i in range(n_metrics)]
    assert opt.eager_analytics_budget_exceeded(_wide_space(n_hp), metrics) is None

    opt.analytics_max_coalitions = expected - 1      # one short
    assert opt.eager_analytics_budget_exceeded(_wide_space(n_hp), metrics)


def test_the_default_budget_admits_both_registry_models():
    """The default has to be useless if it turns away the models that ship."""
    opt = RandomOptimizer()
    assert opt.analytics_max_coalitions == EAGER_MAX_COALITIONS
    four_metrics = ["accuracy", "f1", "precision", "recall(macro)"]
    for n_hp in (4, 6):
        assert opt.eager_analytics_budget_exceeded(_wide_space(n_hp), four_metrics) is None


def test_the_default_budget_turns_away_a_wide_custom_model():
    opt = RandomOptimizer()
    reason = opt.eager_analytics_budget_exceeded(
        _wide_space(10), ["accuracy", "f1", "precision", "recall(macro)"])
    assert reason
    assert "12,288" in reason and "1,024" in reason
    assert "ANALYTICS_EAGER_MAX_COALITIONS" in reason


def test_no_budget_means_no_limit():
    """0 in the setting becomes None here — an explicit "spend whatever"."""
    opt = RandomOptimizer()
    opt.analytics_max_coalitions = None
    assert opt.eager_analytics_budget_exceeded(_wide_space(20), ["accuracy"]) is None


# ── What a skip actually returns ─────────────────────────────────────────────

def test_going_over_budget_skips_every_game_with_the_reason(iris_splits, metrics):
    opt = RandomOptimizer()
    opt.analytics_max_coalitions = 1
    cs = RandomForestModel().get_config_space(seed=0)

    games = opt.compute_hp_games(cs, [], ["accuracy", "f1"], seed=0)

    for game in opt.HP_GAMES:
        importance, warning, interactions = games[game]
        assert importance == {"accuracy": {}, "f1": {}}
        assert interactions == {"accuracy": {}, "f1": {}}
        for m in ("accuracy", "f1"):
            assert "skipped" in warning[m] and "hyperparameters" in warning[m]


def test_a_cancelled_run_skips_the_analytics_entirely():
    """The bug this closes: every optimizer breaks out of its trial loop on
    cancel and then computed the analytics anyway, so Cancel bought you the full
    bill on a run you had just stopped."""
    opt = RandomOptimizer()
    cs = RandomForestModel().get_config_space(seed=0)
    cancelled = threading.Event()
    cancelled.set()

    games = opt.compute_hp_games(cs, [], ["accuracy"], seed=0, cancel_event=cancelled)

    for game in opt.HP_GAMES:
        assert games[game][0] == {"accuracy": {}}
        assert "cancelled" in games[game][1]["accuracy"]
        assert "Resuming" in games[game][1]["accuracy"], "say how to get them back"


def test_cancellation_is_reported_before_the_budget():
    """A cancelled run shouldn't be lectured about a budget it never spent."""
    opt = RandomOptimizer()
    opt.analytics_max_coalitions = 1
    cancelled = threading.Event()
    cancelled.set()

    games = opt.compute_hp_games(
        RandomForestModel().get_config_space(seed=0), [], ["accuracy"],
        seed=0, cancel_event=cancelled)

    assert "cancelled" in games["tunability"][1]["accuracy"]


def test_an_unset_cancel_event_computes_normally(iris_splits, metrics):
    """The guard must key on `is_set()`, not on the event merely existing."""
    X_train, X_val, y_train, y_val = iris_splits
    opt = RandomOptimizer()
    trials = opt.optimize(RandomForestModel(), X_train, y_train, X_val, y_val,
                          metrics=metrics, primary_metric="accuracy",
                          n_trials=3, seed=0).trials
    cs = RandomForestModel().get_config_space(seed=0)

    games = opt.compute_hp_games(cs, trials, ["accuracy"], seed=0,
                                 cancel_event=threading.Event())

    assert games["tunability"][1]["accuracy"] is None
    assert set(games["tunability"][0]["accuracy"]) == set(cs.keys())


# ── End to end through the run service ───────────────────────────────────────

@pytest.mark.django_db
def test_the_run_service_applies_the_deployment_setting(settings):
    """`core/` has no access to Django settings, so the run service sets the
    budget on the optimizer before calling optimize(). Pinning the wiring, since
    a silent failure here means the guard never fires in production."""
    from ui.services.run import create_run, execute_run
    from tests.ui.runs.test_run_execution import _make_experiment

    settings.ANALYTICS_EAGER_MAX_COALITIONS = 1     # nothing is affordable
    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 2}, "accuracy")

    execute_run(run.id)

    exp.refresh_from_db()
    assert exp.result["data"], "trials are still stored"
    assert exp.result["hyperparameter_importance"] == {"accuracy": {}, "f1": {},
                                                      "precision": {},
                                                      "recall(macro)": {}}
    assert "skipped" in exp.result["hyperparameter_importance_warning"]["accuracy"]


@pytest.mark.django_db
def test_zero_in_the_setting_means_no_limit(settings):
    from ui.services.run import create_run, execute_run
    from tests.ui.runs.test_run_execution import _make_experiment

    settings.ANALYTICS_EAGER_MAX_COALITIONS = 0
    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 2}, "accuracy")

    execute_run(run.id)

    exp.refresh_from_db()
    assert exp.result["hyperparameter_importance"]["accuracy"], "computed, not skipped"
