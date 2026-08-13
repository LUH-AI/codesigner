"""Contract tests for the experiment snapshot layer (.ihpo semantics).

These pin the behaviors any experiment store must preserve: what a snapshot
carries, what loading requires, and what survives a save→load cycle. The
database-backed store (Experiment rows + snapshot adapter) must keep every
one of these green.
"""

import json
from importlib.metadata import version as dist_version

import pytest

from core import io
from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

from tests.conftest import DATASETS_DIR, FIXTURES_DIR

# Every top-level key the .ihpo format documents; a store must persist each.
SNAPSHOT_KEYS = (
    "format", "version", "name", "seed", "dataset", "model", "evaluation",
    "metrics", "optimizer", "result",
)

# Snapshot fields that identify the experiment (everything except version and
# format, which the writing app owns, and the dataset path and result, handled
# separately).
IDENTITY_KEYS = (
    ("name",), ("seed",), ("model", "name"), ("model", "path"),
    ("optimizer", "name"), ("optimizer", "params"), ("metrics", "names"),
    ("metrics", "current"), ("metrics", "original"),
)


def _at(snapshot: dict, path: tuple):
    """The value at a dotted path, so a test can name a nested field."""
    for key in path:
        snapshot = snapshot[key]
    return snapshot


def _snapshot(filename: str, dataset_path: str | None = None) -> dict:
    """Parse a bundled fixture, optionally redirecting its dataset_path to a
    file that exists on this machine (the stored paths are machine-specific)."""
    snapshot = io.parse((FIXTURES_DIR / filename).read_bytes())
    if dataset_path is not None:
        snapshot["dataset"]["path"] = dataset_path
    return snapshot


def test_snapshot_carries_all_documented_keys():
    """Both fixtures contain every key the .ihpo format documentation lists.

    A store that persists exactly these keys can represent any experiment.
    """
    for filename in ("test.ihpo", "test2.ihpo"):
        snapshot = _snapshot(filename)
        for key in SNAPSHOT_KEYS:
            assert key in snapshot, f"{filename} lacks {key!r}"


def test_save_load_cycle_preserves_identity_fields(metrics, models, optimizers):
    """Identity fields survive a full load→save cycle unchanged.

    Setup: build a live experiment from the Random Search fixture (dataset
    redirected to a real CSV), then save it and re-parse.
    Expect: every identity field equals the original snapshot's value —
    per the format docs, only the dataset arrays and model object are not
    stored, so nothing else may drift.
    """
    snapshot = _snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers)

    again = io.parse(io.save(name, exp))

    for path in IDENTITY_KEYS:
        assert _at(again, path) == _at(snapshot, path), path


def test_save_stamps_current_version_string():
    """save() writes the running app's version string, not the loaded file's.

    The version field identifies the writer; a file saved by this app must
    carry this app's version.
    """
    snapshot = _snapshot("test2.ihpo")
    exp = {
        "model_name":      snapshot["model"]["name"],
        "model_path":      "",
        "optimizer":       RandomOptimizer(),
        "current_metric":  snapshot["metrics"]["current"],
        "original_metric": snapshot["metrics"]["original"],
        "metrics":         {m: None for m in snapshot["metrics"]["names"]},
        "seed":            snapshot["seed"],
        "dataset_path":    snapshot["dataset"]["path"],
        "result":          None,
    }
    again = io.parse(io.save(snapshot["name"], exp))
    assert again["version"] == dist_version("codesigner")


@pytest.mark.parametrize("filename,opt_cls", [
    ("test.ihpo", SMACOptimizer),      # carries optimizer_state
    ("test2.ihpo", RandomOptimizer),   # base runhistory format
])
def test_result_survives_load_save_cycle(filename, opt_cls, metrics, models, optimizers):
    """A stored result keeps its trials, configs and best score through
    load→save, for both the base format and the SMAC state-carrying format.

    Expect: 30 trials before and after, identical config dicts, identical
    best score, and the reconstructed optimizer has the stored type.
    """
    snapshot = _snapshot(filename, dataset_path=str(DATASETS_DIR / "wine.csv"))
    name, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert isinstance(exp["optimizer"], opt_cls)
    assert len(exp["result"].trials) == 30

    again = io.parse(io.save(name, exp))
    assert again["result"]["configs"] == snapshot["result"]["configs"]
    assert len(again["result"]["data"]) == len(snapshot["result"]["data"])
    assert again["result"]["best_score"] == snapshot["result"]["best_score"]


def test_full_load_requires_dataset_file(metrics, models, optimizers):
    """A full (non-read-only) load fails clearly when the dataset is missing.

    This is the contract behind the re-supply flow: the stored dataset_path
    is machine-specific, and a full load must not proceed without real data.
    """
    snapshot = _snapshot("test2.ihpo")   # stored path doesn't exist here
    with pytest.raises(ValueError, match="dataset not found"):
        io.build_experiment(snapshot, metrics, models, optimizers)


def test_read_only_load_needs_no_files(metrics, models, optimizers):
    """Read-only loading works with no dataset file and yields browsable results.

    Expect: no arrays, but a fully deserialized 30-trial result.
    """
    snapshot = _snapshot("test2.ihpo")
    _, exp = io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)
    assert exp["X_train"] is None
    assert len(exp["result"].trials) == 30


def test_registry_model_substitution_on_load(metrics, models, optimizers):
    """A snapshot naming an unavailable registry model loads under a
    caller-chosen replacement model.

    Expect: without a replacement the load fails; with model_name given, the
    replacement is resolved and recorded as the experiment's model_name.
    """
    snapshot = _snapshot("test2.ihpo")
    snapshot["model"]["name"] = "Discontinued Model"

    with pytest.raises(ValueError, match="not available"):
        io.build_experiment(snapshot, metrics, models, optimizers, read_only=True)

    _, exp = io.build_experiment(
        snapshot, metrics, models, optimizers,
        model_name="SVM Classifier", read_only=True,
    )
    assert exp["model_name"] == "SVM Classifier"
    assert exp["model"] is models["SVM Classifier"]


def test_missing_custom_model_blocks_full_load_but_not_read_only(metrics, models, optimizers):
    """A snapshot with a dead custom-model path can't fully load, but loads
    read-only with model=None.

    A missing model file must block full loads (until re-supplied) while
    read-only browsing stays available.
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
    """Stored optimizer_params reconstruct an equivalent optimizer instance.

    Setup: a snapshot for Grid Search with numeric_steps=7 (non-default).
    Expect: the loaded experiment's optimizer reports the same params, and a
    subsequent save writes them back identically.
    """
    snapshot = _snapshot("test2.ihpo", dataset_path=str(DATASETS_DIR / "wine.csv"))
    snapshot["optimizer"]["name"] = GridOptimizer.name
    snapshot["optimizer"]["params"] = {"numeric_steps": 7}
    snapshot["result"] = None

    name, exp = io.build_experiment(snapshot, metrics, models, optimizers)
    assert isinstance(exp["optimizer"], GridOptimizer)
    assert exp["optimizer"].get_params() == {"numeric_steps": 7}

    again = io.parse(io.save(name, exp))
    assert again["optimizer"]["params"] == {"numeric_steps": 7}


def test_snapshot_json_is_human_readable():
    """Saved .ihpo bytes are indented UTF-8 JSON, as the format documents.

    Human readability is an explicit design goal of the format; the store
    must not switch to compact or binary encoding.
    """
    snapshot = _snapshot("test2.ihpo")
    exp = {
        "model_name":      snapshot["model"]["name"],
        "model_path":      "",
        "optimizer":       RandomOptimizer(),
        "current_metric":  snapshot["metrics"]["current"],
        "original_metric": snapshot["metrics"]["original"],
        "metrics":         {m: None for m in snapshot["metrics"]["names"]},
        "seed":            snapshot["seed"],
        "dataset_path":    "",
        "result":          None,
    }
    text = io.save("readable", exp).decode("utf-8")
    assert json.loads(text)
    assert "\n" in text
