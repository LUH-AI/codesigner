import pytest

from core.models import RandomForestModel
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

from tests.conftest import FakeModel, FakeOptimizer, PreCancelled


def _run(optimizer, model, splits, metrics, n_trials, previous=None, cancel=None):
    """Call optimizer.optimize with the standard argument plumbing.

    Fixed seed 0 and primary_metric "accuracy" keep every test deterministic
    and comparable.
    """
    X_train, X_val, y_train, y_val = splits
    return optimizer.optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy",
        n_trials=n_trials, previous_result=previous, seed=0, cancel_event=cancel,
    )


# ── Random Search ─────────────────────────────────────────────────────────────

def test_random_search_runs_and_numbers_trials(iris_splits, metrics):
    """A fresh random search evaluates exactly n_trials, numbered from 1.

    Also checks the result invariants every optimizer must satisfy: best_score
    equals the max primary-metric score, every trial has a score for every
    metric, and importance is computed per metric.
    """
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)

    assert [t.trial for t in result.trials] == [1, 2, 3]
    assert result.primary_metric == "accuracy"
    assert result.best_score == max(t.scores["accuracy"] for t in result.trials)
    assert set(result.hyperparameter_importance) == set(metrics)
    for t in result.trials:
        assert set(t.scores) == set(metrics)


def test_random_search_resume_continues_numbering(iris_splits, metrics):
    """Resuming a random search extends the history instead of restarting it.

    Setup: a finished 3-trial run passed as previous_result to a 2-trial run.
    Expect: 5 trials numbered 1..5, best score never decreases, and no
    config is evaluated twice (the resume set feeds duplicate-skipping).
    """
    first = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics, n_trials=3)
    combined = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                    n_trials=2, previous=first)

    assert [t.trial for t in combined.trials] == [1, 2, 3, 4, 5]
    assert combined.best_score >= first.best_score
    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(keys) == len(set(keys))


def test_random_search_cancellation(iris_splits, metrics):
    """A cancel flag that is already set stops the run before any trial.

    Uses PreCancelled (a bare object with is_set() → True) to pin down that
    optimizers rely only on the .is_set() contract.
    """
    result = _run(RandomOptimizer(), RandomForestModel(), iris_splits, metrics,
                  n_trials=5, cancel=PreCancelled())
    assert result.trials == []


# ── Grid Search ───────────────────────────────────────────────────────────────

def test_grid_search_reports_grid_size_as_trials_limit(iris_splits, metrics):
    """Grid search exposes the full grid size as trials_limit.

    Setup: numeric_steps=2 over Random Forest's 4 numeric hyperparameters
    → 2⁴ = 16 valid combinations, of which only 4 are run.
    Expect: trials_limit says 16 (the UI uses it to cap the trials slider).
    """
    result = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                  metrics, n_trials=4)

    assert result.trials_limit == 16
    assert [t.trial for t in result.trials] == [1, 2, 3, 4]


def test_grid_search_resume_skips_evaluated_configs(iris_splits, metrics):
    """Resuming a grid search finishes the grid without repeating configs.

    Setup: 4 of 16 grid points evaluated, then a resume asking for 12 more.
    Expect: exactly 16 total trials and 16 distinct configs — the resumed
    run walked only the remaining grid points.
    """
    opt = GridOptimizer(numeric_steps=2)
    first = _run(opt, RandomForestModel(), iris_splits, metrics, n_trials=4)
    combined = _run(GridOptimizer(numeric_steps=2), RandomForestModel(), iris_splits,
                    metrics, n_trials=12, previous=first)

    keys = [tuple(sorted(t.config.items())) for t in combined.trials]
    assert len(combined.trials) == 16
    assert len(set(keys)) == 16


def test_grid_search_get_params_roundtrip():
    """get_params() returns constructor kwargs under their schema names,
    which is how optimizer settings survive a save/load cycle."""
    assert GridOptimizer(numeric_steps=7).get_params() == {"numeric_steps": 7}


# ── Fake optimizer (used by later run-lifecycle tests) ───────────────────────

def test_fake_optimizer_is_instant_and_honours_contract(metrics):
    """FakeOptimizer implements the full optimize() contract it advertises.

    It backs later web-layer run-lifecycle tests, so its behavior is pinned
    here once: fresh runs number from 1, resumes continue numbering and never
    lower the best score, and a pre-set cancel flag yields zero trials.
    """
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
    """SMAC's ask/tell loop runs, records its working dir, and resumes.

    Setup: 3 trials, then 2 more with the first result as previous_result.
    Expect: metadata carries smac_output_dir (needed for state persistence),
    per-metric importance is present, resumed numbering is 1..5, and the
    best score never decreases. Marked slow: real SMAC + model training.
    """
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
    """Serializing a SMAC result embeds the working directory; deserializing
    materializes it again.

    Expect: optimizer_state holds SMAC's files (scenario.json among them),
    each data entry keeps the extension fields (scores, incumbent_score),
    and a deserialized copy points at a restored working dir — the mechanism
    that lets a loaded experiment resume SMAC exactly where it stopped.
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
    assert restored.metadata.get("smac_output_dir")


@pytest.mark.slow
def test_smac_resume_through_serialization_preserves_all_trials(iris_splits, metrics):
    """A resumed SMAC run must not lose trials when persisted between runs.

    Every real run gets a fresh SMACOptimizer instance, and "previous result"
    always arrives via serialize_result/deserialize_result (a saved
    Experiment row or .ihpo file), never the same in-memory object — this is
    the exact round trip resume must survive. (test_smac_runs_and_resumes
    above resumes from the same in-memory object and can't catch this.)
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
