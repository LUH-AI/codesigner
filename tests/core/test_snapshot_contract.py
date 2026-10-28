"""The tests here pin what a snapshot carries and what loading requires;
`tests/experiments/test_snapshot_adapter.py` pins the database round trip.
"""

import json

import pytest

from core import io
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

from tests.conftest import DATASETS_DIR, FIXTURES_DIR

# Top-level .ihpo keys; a store must persist each.
SNAPSHOT_KEYS = (
    "format", "version", "name", "seed", "dataset", "model", "evaluation",
    "metrics", "optimizer", "result",
)

# Identity fields; dataset path and result are handled separately.
IDENTITY_KEYS = (
    ("name",), ("seed",), ("model", "name"), ("model", "path"),
    ("optimizer", "name"), ("optimizer", "params"), ("metrics", "names"),
    ("metrics", "current"), ("metrics", "original"),
)


def _at(snapshot: dict, path: tuple):
    """`_at` returns the value at a nested key path."""
    for key in path:
        snapshot = snapshot[key]
    return snapshot


def _snapshot(filename: str, dataset_path: str | None = None) -> dict:
    """A fixture, with dataset_path optionally pointed at a local file."""
    snapshot = io.parse((FIXTURES_DIR / filename).read_bytes())
    if dataset_path is not None:
        snapshot["dataset"]["path"] = dataset_path
    return snapshot


def test_snapshot_carries_all_documented_keys():
    for filename in ("test.ihpo", "test2.ihpo"):
        snapshot = _snapshot(filename)
        for key in SNAPSHOT_KEYS:
            assert key in snapshot, f"{filename} lacks {key!r}"


@pytest.mark.parametrize("filename,opt_cls", [
    ("test.ihpo", SMACOptimizer),      # optimizer_state
    ("test2.ihpo", RandomOptimizer),   # base format
])
def test_a_stored_result_is_rebuilt_with_its_optimizer(filename, opt_cls, metrics, models,
                                                       optimizers):
    snapshot = _snapshot(filename, dataset_path=str(DATASETS_DIR / "wine.csv"))
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert isinstance(exp["optimizer"], opt_cls)
    assert len(exp["result"].trials) == 30


def test_full_load_requires_dataset_file(metrics, models, optimizers):
    """The re-supply flow relies on a full load refusing a missing dataset.
    """
    snapshot = _snapshot("test2.ihpo")   # path missing here
    with pytest.raises(ValueError, match="dataset not found"):
        io.build_experiment(snapshot, metrics, models, optimizers)


def test_read_only_load_needs_no_files(metrics, models, optimizers):
    snapshot = _snapshot("test2.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    assert exp["X_train"] is None
    assert len(exp["result"].trials) == 30


def test_registry_model_substitution_on_load(metrics, models, optimizers):
    """`build_experiment` resolves `model_name` to a registry copy told the task
    of the file.
    """
    snapshot = _snapshot("test2.ihpo")
    snapshot["model"]["name"] = "Discontinued Model"

    with pytest.raises(ValueError, match="not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    _, exp = io.build_experiment(
        snapshot, metrics, models, optimizers,
        model_name="SVM", read_only=True,
    )
    assert exp["model_name"] == "SVM"
    assert type(exp["model"]) is type(models["SVM"])
    assert exp["model"].task == "classification"


def test_missing_custom_model_blocks_full_load_but_not_read_only(metrics, models, optimizers):
    """A dead custom-model path blocks full loads until re-supplied.
    """
    snapshot = _snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    snapshot["model"]["path"] = "/nonexistent/custom_model.py"

    assert not io.model_path_ok(snapshot)
    with pytest.raises(ValueError, match="custom model error"):
        io.build_experiment(snapshot, metrics, models, optimizers)

    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    assert exp["model"] is None
    assert exp["model_name"] == snapshot["model"]["name"]


def test_optimizer_params_reconstruct_the_optimizer(metrics, models, optimizers):
    """numeric_steps=7 is a non-default value.
    """
    snapshot = _snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    snapshot["optimizer"]["name"] = GridOptimizer.name
    snapshot["optimizer"]["params"] = {"numeric_steps": 7}
    snapshot["result"] = None

    _, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert isinstance(exp["optimizer"], GridOptimizer)
    assert exp["optimizer"].get_params() == {"numeric_steps": 7}


def test_snapshot_json_is_human_readable():
    """Human readability is a design goal of the format.
    """
    text = io.to_bytes(_snapshot("test2.ihpo")).decode("utf-8")
    assert json.loads(text)
    assert "\n" in text
