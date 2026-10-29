"""`SMACOptimizer` must tell SMAC the real budget, because SMAC sizes the
initial design as a share of that budget.
"""

import pytest

from core.optimizers import SMACOptimizer
from core.optimizers.smac import optimizer as smac_optimizer

pytestmark = pytest.mark.slow

INITIAL = "Initial Design"


def _origins(optimizer, result):
    """`_origins` counts initial and model origins."""
    from collections import Counter

    return Counter(
        "initial" if v.startswith(INITIAL) else "model"
        for v in optimizer.serialize_result(result)["config_origins"].values())


def _run(optimizer, models, metrics, iris_splits, trials=30, previous=None):
    X_train, X_val, y_train, y_val = iris_splits
    return optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": trials}, previous_result=previous)


# the regression

def test_a_run_reaches_the_model_rather_than_sampling_throughout(
        models, metrics, iris_splits):
    """A run with only initial-design trials never consulted the surrogate."""
    optimizer = SMACOptimizer()

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits))

    assert counts["model"] > counts["initial"], counts
    assert counts["initial"] > 0, "some exploration is still expected"


def test_the_share_cap_is_what_moves_that_line(models, metrics, iris_splits):
    """A bigger `share_cap` keeps SMAC exploring longer."""
    little = SMACOptimizer(share_cap=0.1)
    plenty = SMACOptimizer(share_cap=0.6)

    few = _origins(little, _run(little, models, metrics, iris_splits))
    many = _origins(plenty, _run(plenty, models, metrics, iris_splits))

    assert many["initial"] > few["initial"], (few, many)


# trial cap

def test_a_trial_cap_is_taken_exactly_by_a_real_search(
        models, metrics, iris_splits):
    """`test_initial_points.py` pins the arithmetic for every cap combination,
    and a real search here samples the computed count."""
    optimizer = SMACOptimizer(trial_cap=7, use_share_cap=False)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=30))

    assert counts["initial"] == 7, counts


# strategies

@pytest.mark.parametrize("strategy", ["gp", "rf"])
def test_either_strategy_runs_and_searches(strategy, models, metrics, iris_splits):
    optimizer = SMACOptimizer(search_strategy=strategy)

    result = _run(optimizer, models, metrics, iris_splits, trials=20)

    assert len(result.trials) == 20
    assert _origins(optimizer, result)["model"] > 0


@pytest.mark.parametrize("strategy", ["gp", "rf"])
def test_either_strategy_answers_when_asked_how_sure_it_is(strategy, models, metrics,
                                                           iris_splits, monkeypatch):
    """The random forest raises on any covariance type except "diagonal", and
    the surrounding `except` would turn the criterion off without an error."""
    X_train, X_val, y_train, y_val = iris_splits
    optimizer = SMACOptimizer(search_strategy=strategy)
    answers = []
    real = smac_optimizer.confidence_nothing_better
    monkeypatch.setattr(
        smac_optimizer, "confidence_nothing_better",
        lambda smac, config_space, incumbent_config:
        answers.append(real(smac, config_space, incumbent_config)) or answers[-1])

    optimizer.optimize(
        models["Random Forest"], X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", seed=0,
        stopping={"max_trials": 20, "incumbent_confidence": 0.999})

    assert answers, "the criterion was never consulted"
    assert all(a is not None for a in answers), strategy
    assert all(0.0 <= a <= 1.0 for a in answers), answers


def test_the_gaussian_process_gives_an_answer_that_actually_varies(
        models, metrics, iris_splits, monkeypatch):
    """The test asserts variation for the Gaussian process only, because a
    forest of the default ten trees gives coarse readings near 0.5."""
    X_train, X_val, y_train, y_val = iris_splits
    seen = {}
    real = smac_optimizer.confidence_nothing_better
    for strategy in ("gp", "rf"):
        optimizer = SMACOptimizer(search_strategy=strategy)
        answers = []
        monkeypatch.setattr(
            smac_optimizer, "confidence_nothing_better",
            lambda smac, config_space, incumbent_config, _s=answers:
            _s.append(real(smac, config_space, incumbent_config)) or _s[-1])
        optimizer.optimize(
            models["Random Forest"], X_train, y_train, X_val, y_val,
            metrics=metrics, primary_metric="accuracy", seed=0,
            stopping={"max_trials": 20, "incumbent_confidence": 0.999})
        seen[strategy] = answers

    assert len(set(seen["gp"])) > 1, seen["gp"]


def test_an_unknown_strategy_falls_back_rather_than_failing(models, metrics, iris_splits):
    """A hand-edited .ihpo can name any strategy."""
    optimizer = SMACOptimizer(search_strategy="quantum")

    assert optimizer.get_params()["search_strategy"] == "gp"
    assert len(_run(optimizer, models, metrics, iris_splits, trials=3).trials) == 3


# resuming

def test_a_resumed_run_does_not_start_exploring_again(models, metrics, iris_splits):
    """SMAC receives the history by replay, and the replay must consume the
    initial design."""
    optimizer = SMACOptimizer()
    first = _run(optimizer, models, metrics, iris_splits, trials=20)

    second = _run(optimizer, models, metrics, iris_splits, trials=10, previous=first)
    added = second.trials[len(first.trials):]

    assert len(added) == 10
    origins = optimizer.serialize_result(second)["config_origins"]
    later = [origins[str(t.trial)] for t in added if str(t.trial) in origins]
    assert later and not all(o.startswith(INITIAL) for o in later), later


def test_changed_settings_take_effect_without_losing_the_history(
        models, metrics, iris_splits):
    first = _run(SMACOptimizer(), models, metrics, iris_splits, trials=10)

    second = _run(SMACOptimizer(search_strategy="rf"), models, metrics,
                  iris_splits, trials=5, previous=first)

    assert len(second.trials) == 15
    assert [t.trial for t in second.trials] == list(range(1, 16))
    assert [t.config for t in second.trials[:10]] == [t.config for t in first.trials]


# surrogate, end to end

def test_a_configured_forest_actually_searches(models, metrics, iris_splits):
    """`test_smac_surrogate` pins the construction."""
    optimizer = SMACOptimizer(search_strategy="rf", rf_trees=64, rf_max_depth=7,
                              rf_feature_ratio=0.5, rf_bootstrapping=False)

    counts = _origins(optimizer, _run(optimizer, models, metrics, iris_splits, trials=14))

    assert counts["model"] > 0, counts


def test_an_mcmc_process_runs_and_is_told_its_trials(models, metrics, iris_splits):
    """MCMC runs three walkers per kernel dimension over 250-step chains."""
    optimizer = SMACOptimizer(search_strategy="gp", gp_model_type="mcmc")

    result = _run(optimizer, models, metrics, iris_splits, trials=3)

    assert len(result.trials) == 3
    assert all(t.scores["accuracy"] > 0 for t in result.trials)
