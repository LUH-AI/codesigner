import pytest

from core.models import RandomForestModel
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

from .conftest import FakeModel, FakeOptimizer, PreCancelled


def _run(optimizer, model, splits, metrics, n_trials, previous=None, cancel=None):
    X_train, X_val, y_train, y_val = splits
    return optimizer.optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy",
        n_trials=n_trials, previous_result=previous, seed=0, cancel_event=cancel,
    )


# ── Random Search ─────────────────────────────────────────────────────────────

def test_random_search_runs_and_numbers_trials(iris_splits, metrics):
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)

    assert [t.trial for t in result.trials] == [1, 2, 3]
    assert result.primary_metric == "accuracy"
    assert result.best_score == max(t.scores["accuracy"] for t in result.trials)
    assert set(result.hyperparameter_importance) == set(metrics)
    for t in result.trials:
        assert set(t.scores) == set(metrics)


def test_random_search_resume_continues_numbering(iris_splits, metrics):
    first = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)
    combined = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=first)

    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]
    assert combined.best_score >= first.best_score
    # resumed run never re-evaluates an already-seen config
    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(keys) == len(set(keys))


def test_random_search_cancellation(iris_splits, metrics):
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                  n_trials=5, cancel=PreCancelled())
    assert result.trials == []


# ── Grid Search ───────────────────────────────────────────────────────────────

def test_grid_search_reports_grid_size_as_trials_limit(iris_splits, metrics):
    result = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                  metrics, n_trials=4)

    # 4 numeric hyperparameters × 2 steps each = 16 valid combinations
    assert result.trials_limit == 16
    assert [t.trial for t in result.trials] == [1, 2, 3, 4]


def test_grid_search_resume_skips_evaluated_configs(iris_splits, metrics):
    opt = GridOptimizer(numeric_steps=2)
    first = _run(opt, RandomForestModel(), iris_splits, metrics, n_trials=4)
    combined = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                    metrics, n_trials=12, previous=first)

    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(combined.trials) == 16
    assert len(set(keys)) == 16


def test_grid_search_get_params_roundtrip():
    assert GridOptimizer(numeric_steps=7).get_params() == {"numeric_steps": 7}


# ── Fake optimizer (used by later run-lifecycle tests) ───────────────────────

def test_fake_optimizer_is_instant_and_honours_contract(metrics):
    model = FakeModel()
    result = FakeOptimizer().optimize(
        model, None, None, None, None,
        metrics=metrics, primary_metric="accuracy", n_trials=10, seed=0,
    )
    assert [t.trial for t in result.trials] == list(range(1, 11))

    resumed = FakeOptimizer().optimize(
        model, None, None, None, None,
        metrics=metrics, primary_metric="accuracy", n_trials=5,
        previous_result=result, seed=1,
    )
    assert [t.trial for t in resumed.trials] == list(range(1, 16))
    assert resumed.best_score >= result.best_score

    cancelled = FakeOptimizer().optimize(
        model, None, None, None, None,
        metrics=metrics, primary_metric="accuracy", n_trials=5,
        cancel_event=PreCancelled(), seed=0,
    )
    assert cancelled.trials == []


# ── SMAC ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_smac_runs_and_resumes(iris_splits, metrics):
    first = _run(SMACOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)

    assert [t.trial for t in first.trials] == [1, 2, 3]
    assert first.metadata.get("smac_output_dir")
    assert set(first.hyperparameter_importance) == set(metrics)

    combined = _run(SMACOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=first)
    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]
    assert combined.best_score >= first.best_score


@pytest.mark.slow
def test_smac_serialize_embeds_optimizer_state(iris_splits, metrics):
    opt = SMACOptimizer()
    result = _run(opt, RandomForestModel(), iris_splits, metrics, n_trials=3)

    d = opt.serialize_result(result)
    assert d["optimizer_state"], "SMAC working directory should be embedded"
    assert any(k.endswith("scenario.json") for k in d["optimizer_state"])
    assert len(d["data"]) == 3
    for entry in d["data"]:
        assert "scores" in entry and "incumbent_score" in entry

    restored = opt.deserialize_result(d)
    assert len(restored.trials) == 3
    assert restored.metadata.get("smac_output_dir")
