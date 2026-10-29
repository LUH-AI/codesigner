"""The page's "Run as Trial" relies on `optimize` evaluating requested
configurations, with their origin, before any search.
"""

import pytest

from core.models import RandomForestModel
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer


def _requested(seed=0):
    """`_requested` sets `max_depth` to 7 so no search would pick the
    configuration first."""
    config_space = RandomForestModel().get_config_space(seed=seed)
    config = dict(config_space.get_default_configuration())
    config["max_depth"] = 7
    return config


def _run(optimizer, splits, metrics, configurations, n_trials=1, previous=None):
    X_train, X_val, y_train, y_val = splits
    return optimizer.optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy",
        n_trials=n_trials, previous_result=previous, seed=0,
        configurations=configurations,
    )


@pytest.mark.parametrize("optimizer", [RandomOptimizer, GridOptimizer])
def test_a_run_of_one_evaluates_exactly_what_it_was_given(optimizer, iris_splits, metrics):
    config = _requested()
    result = _run(optimizer(), iris_splits, metrics,
                  [{"config": config, "origin": "Proposal 7"}])

    assert len(result.trials) == 1
    assert result.trials[0].config == config
    assert result.trials[0].origin == "Proposal 7"


@pytest.mark.parametrize("optimizer", [RandomOptimizer, GridOptimizer])
def test_the_search_carries_on_after_what_was_requested(optimizer, iris_splits, metrics):
    """Requested configurations count against the budget.
    """
    config = _requested()
    result = _run(optimizer(), iris_splits, metrics, [{"config": config}], n_trials=3)

    assert len(result.trials) == 3
    assert result.trials[0].config == config
    assert result.trials[0].origin == "Requested"
    assert all(t.origin != "Requested" for t in result.trials[1:])


def test_without_requests_a_run_searches_as_it_always_did(iris_splits, metrics):
    """`None` and `[]` give the same trials for one seed."""
    before = _run(RandomOptimizer(), iris_splits, metrics, None, n_trials=2)
    after = _run(RandomOptimizer(), iris_splits, metrics, [], n_trials=2)

    assert [t.config for t in before.trials] == [t.config for t in after.trials]


@pytest.mark.slow
def test_smac_evaluates_the_request_and_learns_from_it(iris_splits, metrics):
    """`SMACOptimizer` tells SMAC the requested trial like its own, and the
    runhistory of the resumed run contains all three trials.
    """
    first = _run(SMACOptimizer(), iris_splits, metrics, None, n_trials=2)
    config = _requested()
    result = _run(SMACOptimizer(), iris_splits, metrics,
                  [{"config": config, "origin": "Proposal 3"}], previous=first)

    assert len(result.trials) == 3
    assert result.trials[2].config == config
    assert result.trials[2].origin == "Proposal 3"
    serialized = SMACOptimizer().serialize_result(result)
    assert len(serialized["data"]) == 3
