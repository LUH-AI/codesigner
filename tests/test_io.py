import json

import numpy as np
import pytest

from core import io
from core.optimizers import RandomOptimizer, SMACOptimizer

from .conftest import DATASETS_DIR, FIXTURES_DIR


def _fixture_snapshot(name: str, dataset_path: str | None = None) -> dict:
    snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
    if dataset_path is not None:
        snapshot["dataset_path"] = dataset_path
    return snapshot


# ── parse ─────────────────────────────────────────────────────────────────────

def test_parse_accepts_fixture_files():
    for name in ("test.ihpo", "test2.ihpo"):
        snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
        assert snapshot["name"]
        assert snapshot["metric_names"]


def test_parse_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        io.parse(b"definitely not json {")


def test_parse_rejects_non_string_version():
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["version"] = 2
    with pytest.raises(ValueError, match="unsupported version"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


def test_parse_rejects_missing_field():
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    del snapshot["seed"]
    with pytest.raises(ValueError, match="missing field: 'seed'"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


# ── path checks ───────────────────────────────────────────────────────────────

def test_dataset_path_ok():
    snapshot = _fixture_snapshot("test2.ihpo")
    assert not io.dataset_path_ok(snapshot)          # stored path doesn't exist here
    snapshot["dataset_path"] = str(DATASETS_DIR / "wine.csv")
    assert io.dataset_path_ok(snapshot)


def test_model_path_ok():
    assert io.model_path_ok({"model_path": ""})       # registry model: no file needed
    assert not io.model_path_ok({"model_path": "/nonexistent/model.py"})


# ── splits ────────────────────────────────────────────────────────────────────

def test_load_splits_deterministic_per_seed():
    a = io._load_splits(DATASETS_DIR / "iris.csv", seed=0)
    b = io._load_splits(DATASETS_DIR / "iris.csv", seed=0)
    c = io._load_splits(DATASETS_DIR / "iris.csv", seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], c[0])


def test_load_splits_detects_semicolon_separator(tmp_path):
    csv = tmp_path / "semi.csv"
    rows = ["f1;f2;label"] + [f"{i};{i * 2};{i % 2}" for i in range(20)]
    csv.write_text("\n".join(rows), encoding="utf-8")
    X_train, X_val, y_train, y_val = io._load_splits(csv, seed=0)
    assert X_train.shape[1] == 2
    assert len(X_train) + len(X_val) == 20


def test_attach_dataset(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    io.attach_dataset(exp, str(DATASETS_DIR / "wine.csv"))
    assert exp["X_train"] is not None
    assert len(exp["X_train"]) + len(exp["X_val"]) == len(exp["y_train"]) + len(exp["y_val"])
    assert exp["dataset_path"].endswith("wine.csv")


# ── build_experiment ──────────────────────────────────────────────────────────

def test_build_experiment_read_only(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    assert name == snapshot["name"]
    assert exp["X_train"] is None
    assert isinstance(exp["optimizer"], RandomOptimizer)
    assert exp["model"] is models["Random Forest"]
    assert len(exp["result"].trials) == 30
    assert exp["result"].best_score == snapshot["result"]["best_score"]


def test_build_experiment_smac_restores_optimizer_state(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    assert isinstance(exp["optimizer"], SMACOptimizer)
    assert len(exp["result"].trials) == 30
    smac_dir = exp["result"].metadata.get("smac_output_dir")
    assert smac_dir, "optimizer_state should be materialized to a working directory"


def test_build_experiment_with_dataset(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert exp["X_train"] is not None and exp["y_val"] is not None


def test_build_experiment_unknown_metric(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["metric_names"] = ["accuracy", "nonexistent"]
    with pytest.raises(ValueError, match="unknown metric"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_optimizer(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["optimizer_name"] = "Simulated Annealing"
    with pytest.raises(ValueError, match="optimizer .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_model(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["model_name"] = "Transformer"
    with pytest.raises(ValueError, match="model .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_missing_dataset_raises(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo")   # stored path doesn't exist here
    with pytest.raises(ValueError, match="dataset not found"):
        io.build_experiment(snapshot, metrics, models, optimizers)


# ── save → parse round-trip ───────────────────────────────────────────────────

def test_save_parse_roundtrip(metrics, models, optimizers):
    snapshot = _fixture_snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers)

    again = io.parse(io.save(name, exp))

    for key in ("name", "model_name", "model_path", "optimizer_name",
                "optimizer_params", "primary_metric", "original_metric",
                "metric_names", "seed"):
        assert again[key] == snapshot[key], key
    assert again["result"]["configs"] == snapshot["result"]["configs"]
    assert len(again["result"]["data"]) == len(snapshot["result"]["data"])
    assert again["result"]["best_score"] == snapshot["result"]["best_score"]
