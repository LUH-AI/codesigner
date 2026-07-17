import json
from pathlib import Path

from pytest import approx

from core.optimizers import OptimizationResult, RandomOptimizer, TrialResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _synthetic_result() -> OptimizationResult:
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
    opt = RandomOptimizer()
    original = _synthetic_result()
    restored = opt.deserialize_result(opt.serialize_result(original))

    assert len(restored.trials) == len(original.trials)
    for orig, back in zip(original.trials, restored.trials):
        assert back.trial == orig.trial
        assert back.config == orig.config
        assert back.scores == orig.scores
        # score travels as cost = 1 - score, so allow float round-trip error
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
    opt = RandomOptimizer()
    d = opt.serialize_result(_synthetic_result())

    assert d["stats"] == {"submitted": 3, "finished": 3, "running": 0}
    assert {e["config_id"] for e in d["data"]} == {1, 2, 3}
    # cost is 1 - score, the runhistory convention
    by_id = {e["config_id"]: e for e in d["data"]}
    assert by_id[3]["cost"] == 0.19999999999999996 or abs(by_id[3]["cost"] - 0.2) < 1e-12
    assert d["best_config_id"] == "3"
    assert d["config_origins"] == {"1": "Random Search", "2": "Random Search", "3": "Random Search"}


def test_fixture_result_roundtrips_through_base_format():
    """A stored result dict (no optimizer_state) survives deserialize→serialize."""
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    opt = RandomOptimizer()

    result = opt.deserialize_result(snapshot["result"])
    again = opt.serialize_result(result)

    assert again["configs"] == snapshot["result"]["configs"]
    assert [e["config_id"] for e in again["data"]] == [e["config_id"] for e in snapshot["result"]["data"]]
    for new, old in zip(again["data"], snapshot["result"]["data"]):
        assert abs(new["cost"] - old["cost"]) < 1e-12
        assert new["scores"] == old["scores"]
        assert new["incumbent_config_id"] == old["incumbent_config_id"]
    assert again["best_score"] == snapshot["result"]["best_score"]
    assert again["primary_metric"] == snapshot["result"]["primary_metric"]
