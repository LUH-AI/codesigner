"""Reading a SMAC output directory as an experiment.

*What:* `core/smac_import.py` turns the five files SMAC writes into a snapshot
the rest of the application already understands. Most of it is not translation —
`.ihpo`'s `result` block *is* `runhistory.json` — so what is pinned here is the
handful of places the two genuinely differ, and the edge cases that make the
difference between an import and a plausible-looking fiction.

*How:* against `tests/fixtures/smac_run`, a real 25-trial SMAC run over a
three-hyperparameter space, plus hand-built runhistories for the cases a healthy
run does not contain (repeated configurations, RUNNING rows, infinities,
multi-objective).
"""

import json

import pytest

from core import io
from core.smac_import import (
    NAMESPACE, SmacImportError, snapshot_from_smac,
)

from tests.conftest import FIXTURES_DIR

RUN_DIR = FIXTURES_DIR / "smac_run"
COST = NAMESPACE + "cost"


def _files(**overrides) -> dict:
    """The fixture run's files, with any of them replaced."""
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


# ── the happy path ───────────────────────────────────────────────────────────

def test_a_real_run_becomes_a_readable_snapshot():
    """The whole point: five files in, one snapshot `io.parse` accepts."""
    snapshot = io.parse(io.to_bytes(snapshot_from_smac(_files())))

    assert snapshot["name"] == "demo-run"
    assert snapshot["seed"] == 0
    assert len(snapshot["result"]["data"]) == 25


def test_the_config_space_is_hoisted_rather_than_only_embedded():
    """`configspace.json` becomes the snapshot's `space`, which is what is read.

    It also stays in `optimizer_state`, so a re-export is still a complete copy
    of the directory — but nothing reads it there, which is the gap that made an
    imported run lose every surrogate-backed figure.
    """
    snapshot = snapshot_from_smac(_files())

    space = io.config_space_from_serialized(snapshot["space"], seed=0)
    assert sorted(space.keys()) == ["depth", "kind", "rate"]
    assert "configspace.json" in snapshot["result"]["optimizer_state"]


def test_every_smac_file_is_carried():
    """So a re-export of an imported run is a complete copy of what came in."""
    state = snapshot_from_smac(_files())["result"]["optimizer_state"]

    assert sorted(state) == ["configspace.json", "intensifier.json",
                             "optimization.json", "runhistory.json",
                             "scenario.json"]


def test_the_objective_is_declared_lower_is_better_and_unbounded():
    """The one fact SMAC does not write down, and everything downstream needs.

    A cost has no upper bound to subtract from and no direction stated anywhere
    in the output — so the snapshot declares it, and the declaration travels
    with the numbers rather than in a section that could be separated from them.
    """
    result = snapshot_from_smac(_files())["result"]

    declared = result["declared_metrics"][COST]
    assert declared["higher_is_better"] is False
    assert declared["bounds"] == [None, None]
    assert result["primary_metric"] == COST


def test_the_objective_name_is_namespaced():
    """So its declaration survives a re-export.

    `serialize_result` writes a declaration only for a name the registry does
    not have. An objective called `accuracy` would therefore lose its
    declaration on the first export and read back as a 0-to-1 higher-is-better
    score — a silent inversion, in a file that looks fine.
    """
    scenario = {**_files()["scenario.json"], "objectives": "accuracy"}

    snapshot = snapshot_from_smac(_files(**{"scenario.json": scenario}))

    assert snapshot["metrics"]["names"] == ["smac:accuracy"]
    from core.metrics import METRICS
    assert "smac:accuracy" not in METRICS


def test_the_cost_is_the_score():
    """A lower-is-better metric stores itself unchanged, so the two agree.

    `to_cost`/`from_cost` are the identity for such a metric, so a mismatch here
    would show up as every figure disagreeing with the stored cost.
    """
    for entry in snapshot_from_smac(_files())["result"]["data"]:
        assert entry["scores"][COST] == entry["cost"]


def test_the_incumbent_falls():
    """SMAC stores no trajectory, so one is computed — in the right direction."""
    data = snapshot_from_smac(_files())["result"]["data"]

    scores = [e["incumbent_score"] for e in data]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] == min(e["cost"] for e in data)


def test_it_is_read_only():
    """No model and no dataset, which is what makes it browsable and not runnable.

    A `Scenario` records nothing about what was being optimized, so there is
    nothing to attach a dataset to and nothing that could check one.
    """
    snapshot = snapshot_from_smac(_files())

    assert snapshot["model"]["kind"] == "external"
    assert snapshot["model"]["path"] == ""
    assert snapshot["dataset"]["path"] == ""


# ── the ways a directory differs from a healthy one ──────────────────────────

def test_one_configuration_evaluated_twice_becomes_two_trials():
    """SMAC keys data by (config_id, instance, seed, budget); `.ihpo` by trial.

    `config_id == trial number` is an invariant the whole selection layer
    addresses trials through, so rows are renumbered rather than the invariant
    relaxed — and the key they came from is kept, so nothing is lost.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(config_id=1, cost=0.5, seed=1), _row(config_id=1, cost=0.4, seed=2)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert [e["config_id"] for e in data] == [1, 2]
    assert [e["cost"] for e in data] == [0.5, 0.4]
    assert [e["additional_info"]["smac_key"]["seed"] for e in data] == [1, 2]


def test_a_running_row_is_dropped():
    """Its cost is a placeholder for an answer that never came.

    Imported, it is a trial that never happened reported as one that scored
    2147483647.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=2147483647.0, status=0)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert len(data) == 1
    assert data[0]["cost"] == 0.5


def test_a_trial_with_no_finite_cost_is_dropped():
    """There is no honest score to give it.

    SMAC records `crash_cost` — infinity unless the run set otherwise — for a
    trial that failed. The importer has no dataset, so it cannot compute what a
    model that knew nothing would have scored, and inventing one would put a
    fabricated point in every figure.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=float("inf"), status=2),
         _row(cost=float("nan"), status=2)])})

    data = snapshot_from_smac(files)["result"]["data"]

    assert [e["cost"] for e in data] == [0.5]


def test_a_crash_with_a_real_cost_is_kept_and_marked():
    """A finite crash cost is a measurement of a configuration that does not work.

    Which is a result the search acted on, so it is imported — and marked, so
    the figures draw it as a failure rather than as a bad trial.
    """
    files = _files(**{"runhistory.json": _runhistory(
        [_row(cost=0.5), _row(cost=9.0, status=2)])})

    snapshot = io.parse(io.to_bytes(snapshot_from_smac(files)))
    from ui.registry import OPTIMIZERS
    result = type(OPTIMIZERS["SMAC"])().deserialize_result(snapshot["result"])

    assert [t.failed for t in result.trials] == [False, True]


def test_multi_objective_is_refused_by_name():
    """Rather than silently importing the first component.

    The costs are a trade-off surface; picking one of them would report a search
    for something nobody ran.
    """
    scenario = {**_files()["scenario.json"], "objectives": ["cost", "time"]}

    with pytest.raises(SmacImportError, match="several objectives"):
        snapshot_from_smac(_files(**{"scenario.json": scenario}))


def test_a_directory_with_nothing_finished_is_refused():
    """Rather than producing an experiment with no trials in it."""
    files = _files(**{"runhistory.json": _runhistory([_row(status=0)])})

    with pytest.raises(SmacImportError, match="no finished trials"):
        snapshot_from_smac(files)


def test_a_directory_missing_its_runhistory_is_refused():
    files = _files()
    del files["runhistory.json"]

    with pytest.raises(SmacImportError, match="runhistory.json"):
        snapshot_from_smac(files)


def test_files_are_found_by_basename():
    """So either the run directory or the `smac3_output` above it works.

    A browser's directory picker reports paths relative to whichever folder was
    chosen, and Django keeps only the basename anyway.
    """
    nested = {f"smac3_output/demo-run/0/{name}": content
              for name, content in _files().items()}

    assert len(snapshot_from_smac(nested)["result"]["data"]) == 25


# ── what a file may contain ──────────────────────────────────────────────────

def test_no_infinities_reach_the_file():
    """`scenario.json`'s `crash_cost` defaults to infinity, and is carried.

    `json.dumps` writes that as the bare token `Infinity`, which Python reads
    back and a strict parser refuses — so an `.ihpo` containing one is not the
    portable file it claims to be.
    """
    text = io.to_bytes(snapshot_from_smac(_files())).decode("utf-8")

    assert "Infinity" not in text and "NaN" not in text
    json.loads(text, parse_constant=_refuse)


def test_the_fixture_run_really_does_contain_one():
    """Otherwise the test above proves nothing."""
    scenario = _files()["scenario.json"]

    assert scenario["crash_cost"] == float("inf")


def _refuse(constant):
    raise AssertionError(f"bare {constant} in the file")


# ── the settings the run was configured with ─────────────────────────────────

def test_an_imported_run_keeps_the_settings_it_ran_with():
    """The file always carried these — `_carried` copies `scenario.json`
    verbatim — but under `optimizer_state`, which nothing rebuilds an optimizer
    from. So an imported run displayed as "SMAC" with nothing under it, and
    re-running it would have used codesigner's defaults instead of its own.
    """
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert params["search_strategy"] == "rf"      # HyperparameterOptimizationFacade
    assert params["initial_design"] == "sobol"
    assert params["acquisition"] == "ei"
    assert params["challengers"] == 10000
    assert params["random_probability"] == 0.2
    assert params["retrain_after"] == 8
    assert params["rf_trees"] == 10


def test_an_unbounded_tree_depth_is_not_imported_as_a_limit():
    """SMAC records "no limit" as a real number — 2**20 — so importing it
    literally would put a depth cap of a million on the page and hand it to the
    next run as though someone had chosen it."""
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert "rf_max_depth" not in params


def test_a_setting_that_cannot_be_read_honestly_is_left_out():
    """`max_features` is recorded as a count and `rf_feature_ratio` is a
    fraction; `local_search_iterations` is not in the recorded maximizer at all.
    A guess would show a number on the page the run never used."""
    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert "rf_feature_ratio" not in params
    assert "local_search_iterations" not in params


def test_the_settings_rebuild_an_optimizer():
    """The point of reading them: they have to be accepted by `__init__`. They
    go through `known_params`, so a setting this version no longer has is
    dropped rather than raising from the page that displays the result."""
    from core.optimizers.smac_optimizer import SMACOptimizer

    params = snapshot_from_smac(_files())["optimizer"]["params"]

    assert SMACOptimizer(**params) is not None


def test_a_scenario_with_no_meta_reads_no_settings():
    """Files written before SMAC embedded `_meta`, and anything else that does
    not carry it. Empty rather than guessed."""
    scenario = json.loads((RUN_DIR / "scenario.json").read_text())
    scenario.pop("_meta", None)

    params = snapshot_from_smac(_files(**{"scenario.json": scenario}))["optimizer"]["params"]

    assert params == {}
