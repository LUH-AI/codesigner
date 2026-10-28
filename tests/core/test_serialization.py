import json

from pytest import approx

from core.optimizers import OptimizationResult, RandomOptimizer, TrialResult

from tests.conftest import FIXTURES_DIR


def _synthetic_result() -> OptimizationResult:
    """Trial 2 is non-improving, and trial 3 lists its config keys b before a,
    because key order must leave config identity unchanged."""
    trials = [
        TrialResult(trial=1, config={"a": 1, "b": 2}, scores={"accuracy": 0.5, "f1": 0.4},
                    score=0.5, incumbent_score=0.5, incumbent_config={"a": 1, "b": 2}),
        TrialResult(trial=2, config={"a": 3, "b": 1}, scores={"accuracy": 0.3, "f1": 0.2},
                    score=0.3, incumbent_score=0.5, incumbent_config={"a": 1, "b": 2}),
        TrialResult(trial=3, config={"b": 5, "a": 2}, scores={"accuracy": 0.8, "f1": 0.7},
                    score=0.8, incumbent_score=0.8, incumbent_config={"b": 5, "a": 2}),
    ]
    return OptimizationResult(
        trials=trials,
        primary_metric="accuracy",
        best_config={"b": 5, "a": 2},
        best_score=0.8,
        hyperparameter_importance={"accuracy": {"a": 0.6, "b": 0.4}},
        hyperparameter_importance_warning={"accuracy": None},
        trials_limit=100,
    )


def test_serialize_deserialize_inverse():
    """Scores use approx because they travel as cost = 1 - score.
    """
    opt = RandomOptimizer()
    original = _synthetic_result()
    restored = opt.deserialize_result(opt.serialize_result(original))

    assert len(restored.trials) == len(original.trials)
    for orig, back in zip(original.trials, restored.trials):
        assert back.trial == orig.trial
        assert back.config == orig.config
        assert back.scores == orig.scores
        assert back.score == approx(orig.score)
        assert back.incumbent_score == approx(orig.incumbent_score)
        assert back.incumbent_config == orig.incumbent_config

    assert restored.primary_metric == original.primary_metric
    assert restored.best_config == original.best_config
    assert restored.best_score == original.best_score
    assert restored.hyperparameter_importance == original.hyperparameter_importance
    assert restored.hyperparameter_importance_warning == original.hyperparameter_importance_warning
    assert restored.trials_limit == original.trials_limit


def test_serialized_dict_mirrors_runhistory_shape():
    """`serialize_result` keeps the runhistory shape the .ihpo format promises.
    """
    opt = RandomOptimizer()
    d = opt.serialize_result(_synthetic_result())

    assert d["stats"] == {"submitted": 3, "finished": 3, "running": 0}
    assert {e["config_id"] for e in d["data"]} == {1, 2, 3}
    by_id = {e["config_id"]: e for e in d["data"]}
    assert by_id[3]["cost"] == approx(0.2)
    assert d["best_config_id"] == "3"
    # An unrecorded origin stays empty, because the resume logic tells "unknown"
    # apart from "the model chose it".
    assert d["config_origins"] == {"1": "", "2": "", "3": ""}


def test_fixture_result_roundtrips_through_base_format():
    """test2.ihpo carries no optimizer_state and exercises the base-format path.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    opt = RandomOptimizer()

    result = opt.deserialize_result(snapshot["result"])
    again = opt.serialize_result(result)

    assert again["configs"] == snapshot["result"]["configs"]
    assert [e["config_id"] for e in again["data"]] == [e["config_id"] for e in snapshot["result"]["data"]]
    for new, old in zip(again["data"], snapshot["result"]["data"]):
        assert new["cost"] == approx(old["cost"])
        assert new["scores"] == old["scores"]
        assert new["incumbent_config_id"] == old["incumbent_config_id"]
    assert again["best_score"] == snapshot["result"]["best_score"]
    assert again["primary_metric"] == snapshot["result"]["primary_metric"]
