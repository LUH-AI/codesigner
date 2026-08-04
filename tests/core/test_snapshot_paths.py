"""An experiment file cannot name where its own contents get written.

`optimizer_state` is keyed by the relative path each optimizer file came from,
and deserializing writes those files back out to rebuild the optimizer's state.
Nothing stopped a key from being absolute or containing `..`, and `Path` joining
does not help — an absolute right-hand operand wins outright — so a file edited
by hand could have anything written wherever it liked, as the app user, merely
by being loaded and viewed.

These pin the containment. The load-time check matters as much as the write-time
one: without it a hostile file is accepted and only fails later, whenever
something happens to rebuild its result.
"""

import json
from pathlib import Path

import pytest

from core import io
from core.optimizers import SMACOptimizer
from core.paths import MAX_STATE_FILES, is_safe_relative, safe_join


def _snapshot(optimizer_state):
    return {
        "version": "0.1.0", "name": "hostile", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC (BlackBox)", "optimizer_params": {},
        "primary_metric": "accuracy", "original_metric": "accuracy",
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "",
        "result": {
            "stats": {}, "data": [], "configs": {}, "config_origins": {},
            "optimizer_state": optimizer_state,
            "primary_metric": "accuracy", "best_score": 0.0, "best_config_id": "1",
            "hyperparameter_importance": {}, "hyperparameter_importance_warning": {},
            "trials_limit": None,
        },
    }


# ── The primitive ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "smac3_output/scenario.json",   # what serialization actually produces
    "a/b/c.json",
    "plain.json",
])
def test_relative_names_are_accepted(name):
    assert is_safe_relative(name)


@pytest.mark.parametrize("name", [
    "/etc/passwd",                  # absolute wins the join outright
    "/app/data/db.sqlite3",
    "../../etc/x.json",
    "a/../../../etc/x.json",
    "",
    None,
    42,
])
def test_escaping_names_are_refused(name):
    assert not is_safe_relative(name)


def test_safe_join_keeps_a_name_under_the_root(tmp_path):
    assert safe_join(tmp_path, "smac/scenario.json") == tmp_path / "smac/scenario.json"


def test_safe_join_refuses_an_absolute_name(tmp_path):
    with pytest.raises(ValueError, match="unsafe path"):
        safe_join(tmp_path, "/etc/passwd")


def test_safe_join_refuses_a_symlink_out_of_the_root(tmp_path):
    """A link already under the root is not a way out either — the resolved
    destination is checked, not just the name."""
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="unsafe path"):
        safe_join(tmp_path, "escape/taken.json")


# ── At the boundary: io.parse ────────────────────────────────────────────────

def test_parse_refuses_an_absolute_optimizer_state_key():
    """Refused on load, so a hostile file never reaches a rebuild."""
    data = json.dumps(_snapshot({"/etc/passwd": {"x": 1}})).encode()
    with pytest.raises(ValueError, match="unsafe path"):
        io.parse(data)


def test_parse_refuses_a_traversing_optimizer_state_key():
    data = json.dumps(_snapshot({"../../escaped.json": {"x": 1}})).encode()
    with pytest.raises(ValueError, match="unsafe path"):
        io.parse(data)


def test_parse_caps_the_number_of_state_files():
    state = {f"smac/{i}.json": {} for i in range(MAX_STATE_FILES + 1)}
    with pytest.raises(ValueError, match="more than"):
        io.parse(json.dumps(_snapshot(state)).encode())


def test_parse_accepts_ordinary_state():
    snapshot = _snapshot({"smac3_output/scenario.json": {"name": "x"}})
    assert io.parse(json.dumps(snapshot).encode())["result"]["optimizer_state"]


def test_parse_accepts_a_result_without_optimizer_state():
    """Grid and Random results carry none, and old files may predate it."""
    snapshot = _snapshot({})
    snapshot["result"].pop("optimizer_state")
    assert io.parse(json.dumps(snapshot).encode())


# ── At the write: deserialize_result ─────────────────────────────────────────

def test_deserialize_writes_nothing_outside_its_own_directory(tmp_path):
    """The write-time guard stands on its own, for a result that reached the
    optimizer without passing through parse."""
    target = tmp_path / "written.json"
    with pytest.raises(ValueError, match="unsafe path"):
        SMACOptimizer().deserialize_result(
            _snapshot({str(target): {"x": 1}})["result"])
    assert not target.exists()


def test_deserialize_keeps_a_real_smac_state(tmp_path):
    """The legitimate shape still round-trips: relative keys are written under a
    fresh temp directory and the rebuilt result points at it."""
    result = SMACOptimizer().deserialize_result(_snapshot({
        "smac3_output/scenario.json": {"name": "x", "output_directory": "/old/path"},
        "smac3_output/intensifier.json": {"y": 2},
    })["result"])

    out = Path(result.metadata["smac_output_dir"])
    assert (out / "smac3_output/scenario.json").is_file()
    assert (out / "smac3_output/runhistory.json").is_file()
    # the recorded output directory is rewritten to where the files actually are
    scenario = json.loads((out / "smac3_output/scenario.json").read_text())
    assert scenario["output_directory"] == str(out / "smac3_output")
