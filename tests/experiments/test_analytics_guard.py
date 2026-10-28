"""`compute_hp_games` skips the analytics on cancel, over budget, or with every
display figure off, and explains why in `hyperparameter_*_warning`. Cost is
2^n_hp coalitions per game whatever the trial count.
"""

import threading

import pytest
from ConfigSpace import ConfigurationSpace, Float

from core import analytics
from core.analytics import EAGER_MAX_COALITIONS
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer


def _wide_space(n_hp, seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add([Float(f"h{i}", (0.0, 1.0), default=0.5) for i in range(n_hp)])
    return cs


@pytest.mark.parametrize("n_hp,n_metrics,expected", [
    (4, 4, 192),      # Random Forest, 3.4s measured
    (6, 4, 768),      # SVM, 13s measured
    (8, 4, 3072),
    (10, 4, 12288),   # ~3.5 min, custom upload
])
def test_the_budget_counts_coalitions_across_games_and_metrics(n_hp, n_metrics, expected):
    """The default budget rests on 2^n_hp x 3 games x metrics."""
    metrics = [f"m{i}" for i in range(n_metrics)]
    exceeded = analytics.eager_analytics_budget_exceeded
    assert exceeded(_wide_space(n_hp), metrics, budget=expected) is None    # affordable
    assert exceeded(_wide_space(n_hp), metrics, budget=expected - 1)


def test_the_default_budget_admits_both_registry_models():
    """Shipped models must fit the default budget."""
    assert RandomOptimizer().analytics_max_coalitions == EAGER_MAX_COALITIONS
    four_metrics = ["accuracy", "f1", "precision", "recall(macro)"]
    for n_hp in (4, 6):
        assert analytics.eager_analytics_budget_exceeded(_wide_space(n_hp), four_metrics) is None


def test_the_default_budget_turns_away_a_wide_custom_model():
    reason = analytics.eager_analytics_budget_exceeded(
        _wide_space(10), ["accuracy", "f1", "precision", "recall(macro)"])
    assert reason
    assert "12,288" in reason and "1,024" in reason
    assert "ANALYTICS_EAGER_MAX_COALITIONS" in reason


def test_no_budget_means_no_limit():
    """A setting of 0 arrives here as None."""
    assert analytics.eager_analytics_budget_exceeded(_wide_space(20), ["accuracy"],
                                                     budget=None) is None


def test_going_over_budget_skips_every_game_with_the_reason(iris_splits, metrics):
    cs = RandomForestModel().get_config_space(seed=0)

    games = analytics.compute_hp_games(cs, [], ["accuracy", "f1"], seed=0, budget=1)

    for game in analytics.HP_GAMES:
        importance, warning, interactions, moebius, total = games[game]
        assert importance == {"accuracy": {}, "f1": {}}
        assert interactions == {"accuracy": {}, "f1": {}}
        assert moebius == {"accuracy": [], "f1": []}
        # Zero totals stop the importance figure offering "still to gain" on
        # uncomputed numbers.
        assert total == {"accuracy": 0.0, "f1": 0.0}
        for m in ("accuracy", "f1"):
            reason = warning[m]
            assert reason is not None
            assert "skipped" in reason and "hyperparameters" in reason


def test_a_cancelled_run_skips_the_analytics_entirely():
    cs = RandomForestModel().get_config_space(seed=0)
    cancelled = threading.Event()
    cancelled.set()

    games = analytics.compute_hp_games(cs, [], ["accuracy"], seed=0, cancel_event=cancelled)

    for game in analytics.HP_GAMES:
        assert games[game][0] == {"accuracy": {}}
        reason = games[game][1]["accuracy"]
        assert reason is not None
        assert "cancelled" in reason
        assert "Resuming" in reason, "say how to get them back"


def test_cancellation_is_reported_before_the_budget():
    cancelled = threading.Event()
    cancelled.set()

    games = analytics.compute_hp_games(
        RandomForestModel().get_config_space(seed=0), [], ["accuracy"],
        seed=0, cancel_event=cancelled, budget=1)

    reason = games["tunability"][1]["accuracy"]
    assert reason is not None
    assert "cancelled" in reason


def test_an_unset_cancel_event_computes_normally(iris_splits, metrics):
    """The guard must key on `is_set()`."""
    X_train, X_val, y_train, y_val = iris_splits
    opt = RandomOptimizer()
    trials = opt.optimize(RandomForestModel(), X_train, y_train, X_val, y_val,
                          metrics=metrics, primary_metric="accuracy",
                          n_trials=3, seed=0).trials
    cs = RandomForestModel().get_config_space(seed=0)

    games = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0,
                                       cancel_event=threading.Event())

    assert games["tunability"][1]["accuracy"] is None
    assert set(games["tunability"][0]["accuracy"]) == set(cs.keys())


@pytest.mark.django_db
def test_the_run_service_applies_the_deployment_setting(settings):
    """`core/` cannot read Django settings, so the run service sets the budget."""
    from experiments.services.run import create_run, execute_run
    from tests.experiments.test_run_execution import _make_experiment

    settings.ANALYTICS_EAGER_MAX_COALITIONS = 1     # nothing affordable
    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 2}, "accuracy")

    execute_run(run.id)

    exp.refresh_from_db()
    assert exp.data.result is not None
    assert exp.data.result["data"], "trials are still stored"
    assert exp.data.result["hyperparameter_importance"] == {"accuracy": {}, "f1": {},
                                                      "precision": {},
                                                      "recall(macro)": {}}
    assert "skipped" in exp.data.result["hyperparameter_importance_warning"]["accuracy"]


@pytest.mark.django_db
def test_zero_in_the_setting_means_no_limit(settings):
    from experiments.services.run import create_run, execute_run
    from tests.experiments.test_run_execution import _make_experiment

    settings.ANALYTICS_EAGER_MAX_COALITIONS = 0
    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 2}, "accuracy")

    execute_run(run.id)

    exp.refresh_from_db()
    assert exp.data.result is not None
    assert exp.data.result["hyperparameter_importance"]["accuracy"], "computed, not skipped"


def test_analytics_are_skipped_when_no_figure_will_show_them():
    """Every display figure off leaves the computed fields unread."""
    games = analytics.compute_hp_games(
        RandomForestModel().get_config_space(seed=0), [], ["accuracy"], seed=0, wanted=False)

    for game in analytics.HP_GAMES:
        assert games[game][0] == {"accuracy": {}}
        reason = games[game][1]["accuracy"]
        assert reason is not None
        assert "switched off" in reason


def test_wanting_them_is_the_default_for_a_direct_caller():
    """Direct callers get analytics unless the run service says otherwise."""
    assert RandomOptimizer().analytics_wanted is True


def test_cancellation_is_reported_before_nobody_wanting_them():
    """`compute_hp_games` gives an actionable reason on cancel."""
    cancelled = threading.Event(); cancelled.set()

    games = analytics.compute_hp_games(
        RandomForestModel().get_config_space(seed=0), [], ["accuracy"],
        seed=0, cancel_event=cancelled, wanted=False)

    reason = games["tunability"][1]["accuracy"]
    assert reason is not None
    assert "cancelled" in reason


@pytest.mark.django_db
def test_the_run_service_reads_the_experiments_own_figure_settings():
    """The run service resolves figure settings for `core/`, as for the budget."""
    from experiments.services.run import create_run, execute_run
    from tests.experiments.test_run_execution import _make_experiment

    exp = _make_experiment()
    exp.use_default_settings = False
    from experiments.services.settings import GAME_DISPLAY_FIGURES

    exp.settings = {f"show_{key}": False for key in GAME_DISPLAY_FIGURES}
    exp.save(update_fields=["use_default_settings", "settings"])

    execute_run(create_run(exp, {"max_trials": 2}, "accuracy").id)

    exp.refresh_from_db()
    assert exp.data.result is not None
    assert exp.data.result["data"], "the trials themselves are still stored"
    assert exp.data.result["hyperparameter_importance"]["accuracy"] == {}
    assert "switched off" in exp.data.result["hyperparameter_importance_warning"]["accuracy"]


@pytest.mark.django_db
def test_one_display_figure_left_on_is_enough_to_compute_them():
    """Any one of the six display figures justifies computing the games."""
    from experiments.services.run import create_run, execute_run
    from experiments.services.settings import GAME_DISPLAY_FIGURES
    from tests.experiments.test_run_execution import _make_experiment

    exp = _make_experiment()
    exp.use_default_settings = False
    exp.settings = {f"show_{key}": False for key in GAME_DISPLAY_FIGURES}
    exp.settings["show_interactions_graph"] = True
    exp.save(update_fields=["use_default_settings", "settings"])

    execute_run(create_run(exp, {"max_trials": 2}, "accuracy").id)

    exp.refresh_from_db()
    assert exp.data.result is not None
    assert exp.data.result["hyperparameter_importance"]["accuracy"], "computed, not skipped"
