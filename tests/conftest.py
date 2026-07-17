from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace, Integer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from core.io import _load_splits
from core.models import BaseModel, RandomForestModel, SVMModel
from core.optimizers import (
    BaseOptimizer,
    GridOptimizer,
    OptimizationResult,
    RandomOptimizer,
    SMACOptimizer,
    TrialCollector,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class FakeModel(BaseModel):
    """Instant, deterministic model for exercising optimizer loops in tests.

    The "score" is a pure function of the config, so no training happens and
    results are stable across runs.
    """

    name = "Fake Model"

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("a", (1, 10), default=5), Integer("b", (1, 10), default=5)])
        return cs

    def train_evaluate(self, config, X_train, y_train, X_val, y_val,
                       metrics: dict, seed: int = 0) -> dict:
        score = (int(config["a"]) * int(config["b"])) / 100.0
        return {name: score for name in metrics}


class FakeOptimizer(BaseOptimizer):
    """Instant optimizer for run-lifecycle tests: real TrialCollector bookkeeping
    (resume, incumbent, cancellation) with no actual search or importance math."""

    name = "Fake"
    params_schema = []

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials, previous_result=None, seed: int = 0, cancel_event=None):
        config_space = model.get_config_space(seed=seed)
        collector = TrialCollector(
            target_new_trials=n_trials,
            trial_offset=len(previous_result.trials) if previous_result else 0,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
        )
        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            cfg = dict(config_space.sample_configuration())
            all_scores = model.train_evaluate(
                cfg, X_train, y_train, X_val, y_val, metrics, seed=seed
            )
            collector.record(cfg, all_scores[primary_metric], all_scores)

        all_trials = (previous_result.trials if previous_result else []) + collector.results
        params = list(config_space.keys())
        uniform = {p: 1.0 / len(params) for p in params}
        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=max(all_trials, key=lambda t: t.scores[primary_metric]).config
                        if all_trials else {},
            best_score=max((t.scores[primary_metric] for t in all_trials), default=0.0),
            hyperparameter_importance={m: dict(uniform) for m in metrics},
            hyperparameter_importance_warning={m: None for m in metrics},
        )


class PreCancelled:
    """Minimal cancel flag honouring the .is_set() contract, always cancelled."""

    def is_set(self) -> bool:
        return True


@pytest.fixture
def metrics() -> dict:
    return {
        "accuracy":      lambda y, yp: accuracy_score(y, yp),
        "f1":            lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
        "precision":     lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
        "recall(macro)": lambda y, yp: recall_score(y, yp, average="macro", zero_division=0),
    }


@pytest.fixture
def models() -> dict:
    return {"Random Forest": RandomForestModel(), "SVM Classifier": SVMModel()}


@pytest.fixture
def optimizers() -> dict:
    return {
        "SMAC": SMACOptimizer(),
        "Random Search": RandomOptimizer(),
        "Grid Search": GridOptimizer(),
    }


@pytest.fixture
def iris_splits():
    """(X_train, X_val, y_train, y_val) for datasets/iris.csv with seed 0."""
    return _load_splits(DATASETS_DIR / "iris.csv", seed=0)
