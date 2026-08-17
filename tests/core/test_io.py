import json
from unittest import mock

import numpy as np
import pytest

from core import io
from core.optimizers import RandomOptimizer, SMACOptimizer

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _fixture_snapshot(name: str, dataset_path: str | None = None) -> dict:
    """Parse a fixture .ihpo file, optionally redirecting its dataset_path.

    The fixtures store a dataset path that doesn't exist on this machine;
    tests that need real data point it at a CSV in datasets/ instead.
    """
    snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
    if dataset_path is not None:
        snapshot["dataset"]["path"] = dataset_path
    return snapshot


# ── parse ─────────────────────────────────────────────────────────────────────

def test_parse_accepts_fixture_files():
    """parse() accepts both bundled .ihpo fixtures and returns their fields."""
    for name in ("test.ihpo", "test2.ihpo"):
        snapshot = io.parse((FIXTURES_DIR / name).read_bytes())
        assert snapshot["name"]
        assert snapshot["metrics"]["names"]


def test_parse_rejects_invalid_json():
    """parse() raises ValueError with a readable message on undecodable input."""
    with pytest.raises(ValueError, match="not valid JSON"):
        io.parse(b"definitely not json {")


def test_parse_rejects_non_string_version():
    """parse() rejects files whose version field is not a string.

    Setup: a valid fixture with version overwritten to the integer 2 (the
    shape of pre-string-version files, which this app does not support).
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["version"] = 2
    with pytest.raises(ValueError, match="unsupported version"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


def test_parse_rejects_missing_field():
    """parse() names the missing required field in its error message.

    Setup: a valid fixture with "seed" deleted; the error must say so, since
    that message is shown to the user in the load dialog.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    del snapshot["seed"]
    with pytest.raises(ValueError, match="missing field: 'seed'"):
        io.parse(json.dumps(snapshot).encode("utf-8"))


# ── path checks ───────────────────────────────────────────────────────────────

def test_dataset_path_ok():
    """dataset_path_ok() is False for a missing file, True once the path exists.

    The fixture's stored path is from another machine, so it reports False
    as-is; redirecting to a real CSV flips it.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    assert not io.dataset_path_ok(snapshot)
    snapshot["dataset"]["path"] = str(DATASETS_DIR / "wine.csv")
    assert io.dataset_path_ok(snapshot)


def test_model_path_ok():
    """model_path_ok() is True for registry models (empty path, no file needed)
    and False when a custom-model path points at a missing file."""
    assert io.model_path_ok({"model_path": ""})
    assert not io.model_path_ok({"model_path": "/nonexistent/model.py"})


# ── splits ────────────────────────────────────────────────────────────────────

def test_load_splits_deterministic_per_seed():
    """The train/val split is a pure function of (csv, seed).

    Expect: two calls with seed 0 return identical arrays (this is what lets
    .ihpo files omit the data itself), and seed 1 produces a different split.
    """
    a = io._load_splits(DATASETS_DIR / "iris.csv", seed=0)
    b = io._load_splits(DATASETS_DIR / "iris.csv", seed=0)
    c = io._load_splits(DATASETS_DIR / "iris.csv", seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], c[0])


def test_load_splits_detects_semicolon_separator(tmp_path):
    """CSVs using ';' as separator are sniffed and parsed correctly.

    Setup: a synthetic 20-row, 2-feature semicolon CSV written to tmp_path.
    Expect: 2 feature columns (not 1 unsplit string column) and an 80/20
    row split.
    """
    csv = tmp_path / "semi.csv"
    rows = ["f1;f2;label"] + [f"{i};{i * 2};{i % 2}" for i in range(20)]
    csv.write_text("\n".join(rows), encoding="utf-8")
    X_train, X_val, y_train, y_val = io._load_splits(csv, seed=0)
    assert X_train.shape[1] == 2
    assert len(X_train) + len(X_val) == 20


def test_attach_dataset(metrics, models, optimizers):
    """attach_dataset() fills the experiment's arrays and resolves the path.

    Setup: an experiment built read-only (arrays are None).
    Action: attach wine.csv.
    Expect: train/val arrays populated with consistent X/y sizes, and
    dataset_path updated to the resolved file.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    io.attach_dataset(exp, str(DATASETS_DIR / "wine.csv"))
    assert exp["X_train"] is not None
    assert len(exp["X_train"]) + len(exp["X_val"]) == len(exp["y_train"]) + len(exp["y_val"])
    assert exp["dataset_path"].endswith("wine.csv")


# ── build_experiment ──────────────────────────────────────────────────────────

def test_build_experiment_read_only(metrics, models, optimizers):
    """read_only=True builds a browsable experiment without touching any files.

    Expect: no dataset arrays, but the optimizer is reconstructed with the
    right type, the registry model is resolved, and the stored 30-trial
    result is fully deserialized with its best score intact.
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    assert name == snapshot["name"]
    assert exp["X_train"] is None
    assert isinstance(exp["optimizer"], RandomOptimizer)
    assert exp["model"] is models["Random Forest"]
    assert len(exp["result"].trials) == 30
    assert exp["result"].best_score == snapshot["result"]["best_score"]


def test_build_experiment_smac_carries_optimizer_state(metrics, models, optimizers):
    """Loading a SMAC experiment carries its embedded state on the result.

    The SMAC fixture (test.ihpo) carries optimizer_state (runhistory,
    scenario, intensifier...). Expect: deserialization keeps that dict in
    result.metadata, which is what a later re-serialization passes through and
    what `_pinned_points` reads the initial design out of.

    This used to assert the files were written to a fresh temp dir. They were —
    on every rebuild, and this is a read-only page-render path, so that meant
    one leaked directory per page view. Nothing read them; see
    `SMACOptimizer.deserialize_result`.
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
    """A full (non-read-only) build loads the dataset and produces the splits."""
    snapshot = _fixture_snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert exp["X_train"] is not None and exp["y_val"] is not None


def test_build_experiment_unknown_metric(metrics, models, optimizers):
    """build_experiment() rejects snapshots naming metrics the app doesn't have,
    and the error names the offending metric."""
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["metrics"]["names"] = ["accuracy", "nonexistent"]
    with pytest.raises(ValueError, match="unknown metric"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_optimizer(metrics, models, optimizers):
    """build_experiment() rejects snapshots naming an unavailable optimizer."""
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["optimizer"]["name"] = "Simulated Annealing"
    with pytest.raises(ValueError, match="optimizer .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_unknown_model(metrics, models, optimizers):
    """build_experiment() rejects snapshots naming an unavailable registry model."""
    snapshot = _fixture_snapshot("test2.ihpo")
    snapshot["model"]["name"] = "Transformer"
    with pytest.raises(ValueError, match="model .* not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)


def test_build_experiment_missing_dataset_raises(metrics, models, optimizers):
    """A non-read-only build fails clearly when the stored dataset is absent.

    (The web layer catches this to offer the re-upload / read-only flow.)
    """
    snapshot = _fixture_snapshot("test2.ihpo")
    with pytest.raises(ValueError, match="dataset not found"):
        io.build_experiment(snapshot, metrics, models, optimizers)


# ── save → parse round-trip ───────────────────────────────────────────────────

def test_save_parse_roundtrip(metrics, models, optimizers):
    """An experiment saved and re-parsed preserves every snapshot field.

    Setup: a live experiment built from the Random Search fixture with a
    real dataset.
    Action: save() to bytes, parse() back.
    Expect: all identity/config fields match the original snapshot, and the
    result section keeps its configs, trial count and best score.
    """
    snapshot = _fixture_snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers)

    again = io.parse(io.save(name, exp))

    for path in (("name",), ("seed",), ("model", "name"), ("model", "path"),
                 ("optimizer", "name"), ("optimizer", "params"),
                 ("metrics", "names"), ("metrics", "current"),
                 ("metrics", "original")):
        left, right = again, snapshot
        for key in path:
            left, right = left[key], right[key]
        assert left == right, path
    assert again["result"]["configs"] == snapshot["result"]["configs"]
    assert len(again["result"]["data"]) == len(snapshot["result"]["data"])
    assert again["result"]["best_score"] == snapshot["result"]["best_score"]
