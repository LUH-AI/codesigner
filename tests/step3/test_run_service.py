"""Step 3: the create-and-run service (ui/services/run.py).

Ported from the create-and-run contract validated against InteractiveHPO:
the seed rule and the run contract (trials numbered from 1, all metrics
scored, best_score = max primary score, grid capped at grid size).
"""

import pytest

from ui.services.run import resolve_seed, run_experiment

from tests.conftest import DATASETS_DIR

IRIS = str(DATASETS_DIR / "iris.csv")


def test_resolve_seed_rule():
    """A non-negative seed is used verbatim; a negative one yields a valid random seed."""
    assert resolve_seed(0) == 0
    assert resolve_seed(1234) == 1234
    assert 0 <= resolve_seed(-1) <= 2**31 - 1


def test_run_experiment_produces_valid_result(metrics):
    """A run over iris yields the documented result contract.

    Random Forest + Random Search, 3 trials, accuracy primary.
    Expect: trials numbered 1..3, every metric scored per trial, best_score =
    max primary score, and the primary metric recorded on the result.
    """
    result = run_experiment(
        model_name="Random Forest", optimizer_name="Random Search",
        dataset_path=IRIS, seed=0, primary_metric="accuracy", n_trials=3,
    )
    assert [t.trial for t in result.trials] == [1, 2, 3]
    for t in result.trials:
        assert set(t.scores) == set(metrics)
    assert result.primary_metric == "accuracy"
    assert result.best_score == max(t.scores["accuracy"] for t in result.trials)


def test_run_experiment_is_seed_deterministic():
    """The same seed reproduces the same run; a different seed diverges."""
    kw = dict(model_name="Random Forest", optimizer_name="Random Search",
              dataset_path=IRIS, primary_metric="accuracy", n_trials=3)
    a = run_experiment(seed=0, **kw)
    b = run_experiment(seed=0, **kw)
    c = run_experiment(seed=1, **kw)
    assert [t.config for t in a.trials] == [t.config for t in b.trials]
    assert [t.config for t in a.trials] != [t.config for t in c.trials]


def test_grid_run_caps_at_grid_size():
    """Grid Search over a small grid runs at most the full grid.

    numeric_steps=2 over Random Forest's 4 numeric hyperparameters = 16 configs.
    """
    result = run_experiment(
        model_name="Random Forest", optimizer_name="Grid Search",
        dataset_path=IRIS, seed=0, primary_metric="accuracy", n_trials=100,
        optimizer_params={"numeric_steps": 2},
    )
    assert result.trials_limit == 16
    assert len(result.trials) == 16
