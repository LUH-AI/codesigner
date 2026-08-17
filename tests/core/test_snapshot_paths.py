"""An experiment file cannot name where its own contents get written.

`optimizer_state` is keyed by the relative path each optimizer file came from.
Nothing stopped a key from being absolute or containing `..`, and `Path` joining
does not help — an absolute right-hand operand wins outright — so a file edited
by hand could have anything written wherever it liked, as the app user, merely
by being loaded and viewed.

These pin the containment at both ends. Deserializing no longer writes anything
(the state is carried in memory — see `SMACOptimizer.deserialize_result`), which
removes the traversal *sink* rather than guarding it; but the guards stay, at
parse time and at deserialize time, because the state is still passed through
into whatever gets written next, and a hostile key should be refused where it
enters rather than where it lands.
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

def test_deserialize_refuses_an_unsafe_key_even_though_it_writes_nothing(tmp_path):
    """The guard stands on its own, for a result that reached the optimizer
    without passing through parse.

    Deserializing no longer writes the state out at all (it carries it in
    memory), so traversal is unreachable *here* — but the state is passed
    through into whatever gets serialized next, so a hostile key is still
    refused at the boundary rather than laundered into a new .ihpo.
    """
    target = tmp_path / "written.json"
    with pytest.raises(ValueError, match="unsafe path"):
        SMACOptimizer().deserialize_result(
            _snapshot({str(target): {"x": 1}})["result"])
    assert not target.exists()


def test_deserialize_carries_a_real_smac_state_without_touching_the_disk():
    """The legitimate shape round-trips through memory, not the filesystem.

    This used to assert the opposite — that relative keys were written under a
    fresh temp directory. They were, on every rebuild, and nothing ever deleted
    them; see `deserialize_result`. Nothing needed the files, so the state is
    now carried on the result instead.
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
    """The point of carrying it: a load→save cycle keeps the embedded state, so
    an imported run is still resumable. Without the pass-through this silently
    dropped `optimizer_state`, since there is no live output directory to read a
    runhistory back out of."""
    state = {"smac3_output/scenario.json": {"name": "x"}}
    opt = SMACOptimizer()
    result = opt.deserialize_result(_snapshot(state)["result"])

    assert opt.serialize_result(result)["optimizer_state"] == state
