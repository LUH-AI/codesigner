import json
from typing import cast
from unittest import mock

import numpy as np
import pytest

from core import datasets, io
from core.optimizers import RandomOptimizer, SMACOptimizer

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _fixture_snapshot(name: str, dataset_path: str | None = None) -> dict:
    """Fixtures store a path from another machine, and *dataset_path* points
    them at real data.
    """
    snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
    if dataset_path is not None:
        snapshot["dataset"]["path"] = dataset_path
    return snapshot


def test_parse_accepts_fixture_files():
    for name in ("test.ihpo", "test2.ihpo"):
        snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
        assert snapshot["name"]
        assert snapshot["metrics"]["names"]


def test_parse_rejects_invalid_json():
    """The message stays readable."""
    with pytest.raises(ValueError, match="not valid JSON"):
        io.parse(b"definitely not json {")


def test_parse_rejects_non_string_version():
    """The app rejects integer versions from pre-string files.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["version"] = 2
    with pytest.raises(ValueError, match="unsupported version"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


def test_parse_rejects_missing_field():
    """`io.parse` names the missing field, which the load dialog shows the user.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    del snapshot["seed"]
    with pytest.raises(ValueError, match="missing field: 'seed'"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


def test_dataset_path_ok():
    """The fixture's path comes from another machine until redirected.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    assert not io.dataset_path_ok(snapshot)
    snapshot["dataset"]["path"] = str(DATASETS_DIR / "wine.csv")
    assert io.dataset_path_ok(snapshot)


def test_model_path_ok():
    """Registry models need no file; a missing custom path fails."""
    assert io.model_path_ok({"model_path": ""})
    assert not io.model_path_ok({"model_path": "/nonexistent/model.py"})


def test_load_splits_deterministic_per_seed():
    """Seeded determinism lets `.ihpo` files omit the data.
    """
    a = datasets.load_splits(DATASETS_DIR / "iris.csv", seed=0)
    b = datasets.load_splits(DATASETS_DIR / "iris.csv", seed=0)
    c = datasets.load_splits(DATASETS_DIR / "iris.csv", seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], c[0])


def test_load_splits_detects_semicolon_separator(tmp_path):
    """A semicolon CSV must split into two feature columns.
    """
    csv = tmp_path / "semi.csv"
    rows = ["f1;f2;label"] + [f"{i};{i * 2};{i % 2}" for i in range(20)]
    csv.write_text("\n".join(rows), encoding="utf-8")
    X_train, X_val, y_train, y_val = datasets.load_splits(csv, seed=0)
    assert cast(np.ndarray, X_train).shape[1] == 2
    assert len(X_train) + len(X_val) == 20


def test_attach_dataset(metrics, models, optimizers):
    """`attach_dataset` fills a read-only build's arrays and resolves the path.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    io.attach_dataset(exp, str(DATASETS_DIR / "wine.csv"))
    assert exp["X_train"] is not None
    assert len(exp["X_train"]) + len(exp["X_val"]) == len(exp["y_train"]) + len(exp["y_val"])
    assert exp["dataset_path"].endswith("wine.csv")


def test_build_experiment_read_only(metrics, models, optimizers):
    """The model is a copy told the file's task, since every experiment shares
    the registry instance.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    assert name == snapshot["name"]
    assert exp["X_train"] is None
    assert isinstance(exp["optimizer"], RandomOptimizer)
    assert type(exp["model"]) is type(models["Random Forest"])
    assert exp["model"] is not models["Random Forest"]
    assert exp["model"].task == "classification"
    assert len(exp["result"].trials) == 30
    assert exp["result"].best_score == snapshot["result"]["best_score"]


def test_build_experiment_smac_carries_optimizer_state(metrics, models, optimizers):
    """The result metadata keeps `optimizer_state`, which `_pinned_points`
    reads, and rebuilding writes nothing to disk.
    """
    snapshot = _fixture_snapshot("test.ihpo")

    with mock.patch("tempfile.mkdtemp", side_effect=AssertionError("wrote to disk")):
        _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    assert isinstance(exp["optimizer"], SMACOptimizer)
    assert len(exp["result"].trials) == 30
    state = exp["result"].metadata.get("optimizer_state")
    assert state, "optimizer_state should be carried on the rebuilt result"
    assert any(key.endswith("scenario.json") for key in state)


def test_build_experiment_with_dataset(metrics, models, optimizers):
    """A full build loads the dataset."""
    snapshot = _fixture_snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert exp["X_train"] is not None and exp["y_val"] is not None


def test_build_experiment_unknown_metric(metrics, models, optimizers):
    """`build_experiment` names the unknown metric in its error."""
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["metrics"]["names"] = ["accuracy", "nonexistent"]
    with pytest.raises(ValueError, match="unknown metric"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_optimizer(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["optimizer"]["name"] = "Simulated Annealing"
    with pytest.raises(ValueError, match="optimizer .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_model(metrics, models, optimizers):
    """Unknown registry models fail too."""
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["model"]["name"] = "Transformer"
    with pytest.raises(ValueError, match="model .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_missing_dataset_raises(metrics, models, optimizers):
    """The web layer catches the error and offers re-upload or a read-only view.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    with pytest.raises(ValueError, match="dataset not found"):
        io.build_experiment(snapshot, metrics, models, optimizers)
