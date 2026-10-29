import pytest

from core.models import RandomForestModel
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

from tests.conftest import FakeModel, FakeOptimizer, PreCancelled


def _run(optimizer, model, splits, metrics, n_trials, previous=None, cancel=None, seed=0):
    """Seed 0 and primary metric accuracy keep every test deterministic.
    """
    X_train, X_val, y_train, y_val = splits
    return optimizer.optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy",
        n_trials=n_trials, previous_result=previous, seed=seed, cancel_event=cancel,
    )


def test_random_search_runs_and_numbers_trials(iris_splits, metrics):
    """Every optimizer's result keeps best_score at the max primary score, a
    score per metric, and per-metric importance.
    """
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)

    assert [t.trial for t in result.trials] == [1, 2, 3]
    assert result.primary_metric == "accuracy"
    assert result.best_score == max(t.scores["accuracy"] for t in result.trials)
    assert set(result.hyperparameter_importance) == set(metrics)
    for t in result.trials:
        assert set(t.scores) == set(metrics)


def test_random_search_is_seed_deterministic(iris_splits, metrics):
    """Different seeds draw different configurations."""
    a, b, c = (_run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=3, seed=seed) for seed in (0, 0, 1))
    assert [t.config for t in a.trials] == [t.config for t in b.trials]
    assert [t.config for t in a.trials] != [t.config for t in c.trials]


def test_random_search_resume_continues_numbering(iris_splits, metrics):
    """A resume numbers trials 1..5, never lowers the best score, and skips
    evaluated configs.
    """
    first = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)
    combined = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=first)

    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]
    assert combined.best_score is not None and first.best_score is not None
    assert combined.best_score >= first.best_score
    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(keys) == len(set(keys))


def test_random_search_cancellation(iris_splits, metrics):
    """`PreCancelled` proves optimizers only rely on `.is_set()`.
    """
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                  n_trials=5, cancel=PreCancelled())
    assert result.trials == []


def test_grid_search_reports_grid_size_as_trials_limit(iris_splits, metrics):
    """numeric_steps=2 over four numeric hyperparameters gives a 16-point grid,
    and the UI caps the trials slider at `trials_limit`.
    """
    result = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                  metrics, n_trials=4)

    assert result.trials_limit == 16
    assert [t.trial for t in result.trials] == [1, 2, 3, 4]


def test_grid_search_resume_skips_evaluated_configs(iris_splits, metrics):
    """The resume only walks the 12 remaining grid points.
    """
    opt = GridOptimizer(numeric_steps=2)
    first = _run(opt, RandomForestModel(), iris_splits, metrics, n_trials=4)
    combined = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                    metrics, n_trials=12, previous=first)

    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(combined.trials) == 16
    assert len(set(keys)) == 16


def test_grid_search_get_params_roundtrip():
    """Optimizer settings survive save and load through schema-named kwargs."""
    assert GridOptimizer(numeric_steps=7).get_params() == {"numeric_steps": 7}


def test_fake_optimizer_is_instant_and_honours_contract(metrics, tiny_splits):
    """Later run-lifecycle tests rely on `FakeOptimizer` honouring numbering,
    resume and cancellation.
    """
    model = FakeModel()
    X_train, X_val, y_train, y_val = tiny_splits
    result = FakeOptimizer().optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=10, seed=0,
    )
    assert [t.trial for t in result.trials] == list(range(1, 11))

    resumed = FakeOptimizer().optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5,
        previous_result=result, seed=1,
    )
    assert [t.trial for t in resumed.trials] == list(range(1, 16))
    assert resumed.best_score is not None and result.best_score is not None
    assert resumed.best_score >= result.best_score

    cancelled = FakeOptimizer().optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5,
        cancel_event=PreCancelled(), seed=0,
    )
    assert cancelled.trials == []


@pytest.mark.slow
def test_smac_runs_and_resumes(iris_splits, metrics):
    """SMAC records `smac_output_dir`, computes per-metric importance, and
    resumes with numbering 1..5.
    """
    first = _run(SMACOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)

    assert [t.trial for t in first.trials] == [1, 2, 3]
    assert first.metadata.get("smac_output_dir")
    assert set(first.hyperparameter_importance) == set(metrics)

    combined = _run(SMACOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=first)
    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]
    assert combined.best_score is not None and first.best_score is not None
    assert combined.best_score >= first.best_score


@pytest.mark.slow
def test_smac_serialize_embeds_optimizer_state(iris_splits, metrics):
    """`optimizer_state` carries SMAC's files, letting a loaded experiment
    resume where SMAC stopped. Deserializing keeps the state in memory and
    writes no files.
    """
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
    assert restored.metadata.get("optimizer_state") == d["optimizer_state"]


@pytest.mark.slow
def test_smac_resume_through_serialization_preserves_all_trials(iris_splits, metrics):
    """Real resumes arrive through serialize and deserialize on a fresh
    `SMACOptimizer`, a path the in-memory resume test misses.
    """
    opt1 = SMACOptimizer()
    first = _run(opt1, RandomForestModel(), iris_splits, metrics, n_trials=3)
    snapshot1 = opt1.serialize_result(first)
    assert len(snapshot1["data"]) == 3

    opt2 = SMACOptimizer()
    restored = opt2.deserialize_result(snapshot1)
    combined = _run(opt2, RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=restored)
    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]

    snapshot2 = opt2.serialize_result(combined)
    assert [e["config_id"] for e in snapshot2["data"]] == [1, 2, 3, 4, 5], (
        "serialize_result must reflect every trial after a resume, not only "
        "the newly-run ones"
    )
