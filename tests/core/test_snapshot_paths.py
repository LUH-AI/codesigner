"""`optimizer_state` keys are relative paths, and an absolute or `..` key in a
hand-edited file could point a write anywhere. `io.parse` and
`SMACOptimizer.deserialize_result` both refuse such keys where they enter.
"""

import json
from pathlib import Path
from unittest import mock

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


@pytest.mark.parametrize("name", [
    "smac3_output/scenario.json",   # serialized shape
    "a/b/c.json",
    "plain.json",
])
def test_relative_names_are_accepted(name):
    assert is_safe_relative(name)


@pytest.mark.parametrize("name", [
    "/etc/passwd",                  # absolute replaces root
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
    """`safe_join` checks the resolved destination, so a symlink under the root
    cannot escape."""
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="unsafe path"):
        safe_join(tmp_path, "escape/taken.json")


def test_parse_refuses_an_absolute_optimizer_state_key():
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
    """Grid, Random and older results carry none."""
    snapshot = _snapshot({})
    snapshot["result"].pop("optimizer_state")
    assert io.parse(json.dumps(snapshot).encode())


def test_deserialize_refuses_an_unsafe_key_even_though_it_writes_nothing(tmp_path):
    """`deserialize_result` checks keys itself, for a result that skipped
    `io.parse`. The carried state reaches the next save, and a hostile key would
    end up in a new .ihpo.
    """
    target = tmp_path / "written.json"
    with pytest.raises(ValueError, match="unsafe path"):
        SMACOptimizer().deserialize_result(
            _snapshot({str(target): {"x": 1}})["result"])
    assert not target.exists()


def test_deserialize_carries_a_real_smac_state_without_touching_the_disk():
    """`deserialize_result` keeps the state in memory and writes no temp files.
    """
    state = {
        "smac3_output/scenario.json": {"name": "x", "output_directory": "/old/path"},
        "smac3_output/intensifier.json": {"y": 2},
    }
    with mock.patch("tempfile.mkdtemp", side_effect=AssertionError("wrote to disk")):
        result = SMACOptimizer().deserialize_result(_snapshot(state)["result"])

    assert result.metadata["optimizer_state"] == state
    assert "smac_output_dir" not in result.metadata


def test_carried_state_survives_re_serialization():
    """`serialize_result` writes carried state back out, and an imported run
    stays resumable after a load and save."""
    state = {"smac3_output/scenario.json": {"name": "x"}}
    opt = SMACOptimizer()
    result = opt.deserialize_result(_snapshot(state)["result"])

    assert opt.serialize_result(result)["optimizer_state"] == state
