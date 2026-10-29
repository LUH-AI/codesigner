"""`core/optimizers/smac/importer.py` turns the five files of a SMAC output
directory into a snapshot. The tests pin where `.ihpo` and SMAC differ, against
`tests/fixtures/smac_run` and hand-built runhistories.
"""

import json

import pytest

from core import io
from core.optimizers.smac.importer import (
    NAMESPACE, SmacImportError, snapshot_from_smac,
)

from tests.conftest import FIXTURES_DIR

RUN_DIR = FIXTURES_DIR / "smac_run"
COST = NAMESPACE + "cost"


def _files(**overrides) -> dict:
    """`_files` loads fixture files with overrides."""
    files = {p.name: json.loads(p.read_text()) for p in RUN_DIR.glob("*.json")}
    files.update(overrides)
    return files


def _runhistory(data, configs=None, origins=None) -> dict:
    return {"stats": {"submitted": len(data), "finished": len(data), "running": 0},
            "data": data,
            "configs": configs or {"1": {"depth": 3, "rate": 0.2, "kind": "a"}},
            "config_origins": origins or {}}


def _row(config_id=1, cost=0.5, status=1, **extra):
    return {"config_id": config_id, "instance": None, "seed": 0, "budget": None,
            "cost": cost, "time": 0.1, "cpu_time": 0.1, "status": status,
            "starttime": 1.0, "endtime": 1.1, "additional_info": {}, **extra}


# happy path

def test_a_real_run_becomes_a_readable_snapshot():
    snapshot = io.parse(io.to_bytes(snapshot_from_smac(_files())))

    assert snapshot["name"] == "demo-run"
    assert snapshot["seed"] == 0
    assert len(snapshot["result"]["data"]) == 25


def test_the_config_space_is_hoisted_rather_than_only_embedded():
    """Surrogate-backed figures read the config space from `space`, and nothing
    reads the copy in `optimizer_state`.
    """
    snapshot = snapshot_from_smac(_files())

    space = io.config_space_from_serialized(snapshot["space"], seed=0)
    assert space is not None
    assert sorted(space.keys()) == ["depth", "kind", "rate"]
    assert "configspace.json" in snapshot["result"]["optimizer_state"]


def test_every_smac_file_is_carried():
    state = snapshot_from_smac(_files())["result"]["optimizer_state"]

    assert sorted(state) == ["configspace.json", "intensifier.json",
                             "optimization.json", "runhistory.json",
                             "scenario.json"]


def test_the_objective_is_declared_lower_is_better_and_unbounded():
    """The snapshot declares direction and bounds beside the numbers, because
    SMAC output states neither for a cost.
    """
    result = snapshot_from_smac(_files())["result"]

    declared = result["declared_metrics"][COST]
    assert declared["higher_is_better"] is False
    assert declared["bounds"] == [None, None]
    assert result["primary_metric"] == COST


def test_the_objective_name_is_namespaced():
    """`serialize_result` only declares unregistered names, and a bare
    `accuracy` would read back as higher-is-better.
    """
    scenario = {**_files()["scenario.json"], "objectives": "accuracy"}

    snapshot = snapshot_from_smac(_files(**{"scenario.json": scenario}))

    assert snapshot["metrics"]["names"] == ["smac:accuracy"]
    from core.metrics import METRICS
    assert "smac:accuracy" not in METRICS


def test_the_cost_is_the_score():
    """`to_cost` and `from_cost` are the identity for a lower-is-better metric.
    """
    for entry in snapshot_from_smac(_files())["result"]["data"]:
        assert entry["scores"][COST] == entry["cost"]


def test_the_incumbent_falls():
    """SMAC stores no trajectory, so the importer computes one."""
    data = snapshot_from_smac(_files())["result"]["data"]

    scores = [e["incumbent_score"] for e in data]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] == min(e["cost"] for e in data)


def test_it_is_read_only():
    """A `Scenario` records nothing to attach a dataset to.
    """
    snapshot = snapshot_from_smac(_files())

    assert snapshot["model"]["kind"] == "external"
    assert snapshot["model"]["path"] == ""
    assert snapshot["dataset"]["path"] == ""


# unhealthy directories

def test_one_configuration_evaluated_twice_becomes_two_trials():
    """`snapshot_from_smac` renumbers rows and keeps the SMAC key, because
    trials are addressed by `config_id == trial number`.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(config_id=1, cost=0.5, seed=1), _row(config_id=1, cost=0.4, seed=2)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert [e["config_id"] for e in data] == [1, 2]
    assert [e["cost"] for e in data] == [0.5, 0.4]
    assert [e["additional_info"]["smac_key"]["seed"] for e in data] == [1, 2]


def test_a_running_row_is_dropped():
    """The cost of a running row is a placeholder.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=2147483647.0, status=0)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert len(data) == 1
    assert data[0]["cost"] == 0.5


def test_a_trial_with_no_finite_cost_is_dropped():
    """The importer has no dataset to compute a fallback score from.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=float("inf"), status=2),
         _row(cost=float("nan"), status=2)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert [e["cost"] for e in data] == [0.5]


def test_a_crash_with_a_real_cost_is_kept_and_marked():
    """A finite crash cost is a result the search acted on, and the figures draw
    the trial as a failure.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=9.0, status=2)])})

    snapshot = io.parse(io.to_bytes(snapshot_from_smac(files)))
    from core.registry import OPTIMIZERS
    result = OPTIMIZERS["SMAC"]().deserialize_result(snapshot["result"])

    assert [t.failed for t in result.trials] == [False, True]


def test_multi_objective_is_refused_by_name():
    """Picking one cost would report a search nobody ran.
    """
    scenario = {**_files()["scenario.json"], "objectives": ["cost", "time"]}

    with pytest.raises(SmacImportError, match="several objectives"):
        snapshot_from_smac(_files(**{"scenario.json": scenario}))


def test_a_directory_with_nothing_finished_is_refused():
    files = _files(**{"runhistory.json": _runhistory([_row(status=0)])})

    with pytest.raises(SmacImportError, match="no finished trials"):
        snapshot_from_smac(files)


def test_a_directory_missing_its_runhistory_is_refused():
    files = _files()
    del files["runhistory.json"]

    with pytest.raises(SmacImportError, match="runhistory.json"):
        snapshot_from_smac(files)


def test_files_are_found_by_basename():
    """Browser directory pickers report relative paths, and Django only keeps
    basenames.
    """
    nested = {f"smac3_output/demo-run/0/{name}": content
              for name, content in _files().items()}

    assert len(snapshot_from_smac(nested)["result"]["data"]) == 25


# file contents

def test_no_infinities_reach_the_file():
    """`json.dumps` writes the infinite default `crash_cost` as a bare
    `Infinity`, which strict JSON parsers fail to read.
    """
    text = io.to_bytes(snapshot_from_smac(_files())).decode("utf-8")

    assert "Infinity" not in text and "NaN" not in text
    json.loads(text, parse_constant=_refuse)


def test_the_fixture_run_really_does_contain_one():
    """The fixture check guards the test above."""
    scenario = _files()["scenario.json"]

    assert scenario["crash_cost"] == float("inf")


def _refuse(constant):
    raise AssertionError(f"bare {constant} in the file")


# run settings

def test_an_imported_run_keeps_the_settings_it_ran_with():
    """`snapshot_from_smac` reads the settings into `params`, because nothing
    rebuilds an optimizer from `optimizer_state`.
    """
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert params["search_strategy"] == "rf"
    assert params["initial_design"] == "sobol"
    assert params["acquisition"] == "ei"
    assert params["challengers"] == 10000
    assert params["random_probability"] == 0.2
    assert params["retrain_after"] == 8
    assert params["rf_trees"] == 10


def test_an_unbounded_tree_depth_is_not_imported_as_a_limit():
    """SMAC records "no limit" as 2**20, which would pass to the next run as a
    chosen cap."""
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert "rf_max_depth" not in params


def test_a_setting_that_cannot_be_read_honestly_is_left_out():
    """`max_features` is a count while `rf_feature_ratio` is a fraction, and the
    recorded maximizer lacks `local_search_iterations`."""
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert "rf_feature_ratio" not in params
    assert "local_search_iterations" not in params


def test_the_settings_rebuild_an_optimizer():
    """`known_params` drops settings this version lacks before they reach
    `__init__`."""
    from core.optimizers.smac import SMACOptimizer

    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert SMACOptimizer(**params) is not None


def test_a_scenario_with_no_meta_reads_no_settings():
    """Files without `_meta` read as empty settings."""
    scenario = json.loads((RUN_DIR / "scenario.json").read_text())
    scenario.pop("_meta", None)

    params = snapshot_from_smac(_files(**{"scenario.json": scenario}))["optimizer"]["params"]

    assert params == {}
