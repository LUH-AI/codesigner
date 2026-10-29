import json
from pathlib import Path

from ConfigSpace import Configuration
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from ...metrics import metric_for, to_cost
from ...paths import is_safe_relative
from ..base import OptimizationResult, storable_score
from ..timing import STATUS_SUCCESS


def trial_value(cost: float, run_info: dict) -> TrialValue:
    """SMAC's runhistory, copied by `serialized`, gets the measured timing and
    the real status, so the surrogate learns from failures."""
    return TrialValue(
        cost=cost,
        time=run_info.get("time", 0.0), cpu_time=run_info.get("cpu_time", 0.0),
        starttime=run_info.get("starttime", 0.0), endtime=run_info.get("endtime", 0.0),
        status=StatusType(run_info.get("status", STATUS_SUCCESS)),
        # The reason tells a bad configuration from a broken model.
        additional_info=run_info.get("additional_info") or {},
    )


def replay(smac, config_space, trials, primary_metric: str, seed: int,
           replay_metric=None) -> None:
    """`replay` tells SMAC an earlier run's trials when stored state cannot
    resume, after a metric change or a lost directory. `save=False` skips
    rewriting the runhistory per replayed trial, since the first real trial
    saves the file.
    """
    replay_metric = replay_metric or metric_for(primary_metric)
    for t in trials:
        try:
            config = Configuration(config_space, values=t.config)
        except Exception:
            # The search space changed, so the configuration cannot be told.
            continue
        # SMAC stamps a told configuration that lacks an origin as Custom,
        # so the stored origin goes back first, with None when unknown.
        config.origin = t.origin or None
        smac.tell(
            TrialInfo(config=config, seed=seed),
            # `to_cost` gives older metrics `1.0 - score`, and a replay
            # therefore tells SMAC the first run's costs.
            trial_value(to_cost(replay_metric, t.scores.get(primary_metric, t.score)),
                        t.run_info),
            save=False,
        )


def serialized(out: dict, result: OptimizationResult) -> dict:
    """SMAC's runhistory and output directory replace the base trial table
    when present."""
    output_dir = result.metadata.get("smac_output_dir", "")
    root = Path(output_dir) if output_dir else None
    rh_files = list(root.rglob("runhistory.json")) if root and root.exists() else []
    if root is None or not rh_files:
        # A result rebuilt by `restored` carries its state in `metadata`, and
        # passing that through lets an imported run resume after a save.
        carried = result.metadata.get("optimizer_state") or {}
        return {**out, "optimizer_state": carried} if carried else out

    rh_path = rh_files[0]
    rh = json.loads(rh_path.read_text(encoding="utf-8"))

    trial_by_id = {t.trial: t for t in result.trials}
    config_key_to_id = {
        tuple(sorted(t.config.items())): str(t.trial) for t in result.trials
    }

    # SMAC's runhistory can hold entries the run never recorded, such as a
    # trial asked for before a stop, and those would appear as unscored
    # trials.
    recorded = []
    for entry in rh["data"]:
        t = trial_by_id.get(entry["config_id"])
        if t is None:
            continue
        entry["scores"] = t.scores
        entry["incumbent_score"] = storable_score(t.incumbent_score)
        incumbent_key = tuple(sorted(t.incumbent_config.items()))
        entry["incumbent_config_id"] = int(
            config_key_to_id.get(incumbent_key, str(entry["config_id"]))
        )
        recorded.append(entry)

    return {
        **out,
        # Counts follow the kept entries.
        "stats": {"submitted": len(recorded), "finished": len(recorded), "running": 0},
        "data": recorded,
        "configs": rh["configs"],
        "optimizer_state": {
            f.relative_to(root).as_posix(): json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(root.rglob("*"))
            if f.is_file() and f != rh_path
        },
    }


def restored(result: OptimizationResult, optimizer_state: dict) -> OptimizationResult:
    """The SMAC state stays in `metadata`, since a resume makes its own
    directory and replays history through `tell`. State keys are validated
    anyway, so a bad path cannot pass through into the next `.ihpo`.
    """
    if not optimizer_state:
        return result

    for rel in optimizer_state:
        if not is_safe_relative(rel):
            raise ValueError(f"unsafe path in experiment file: {rel!r}")

    first_key = next(iter(optimizer_state))
    subdir = "/".join(first_key.split("/")[:-1])

    result.metadata["optimizer_state"] = optimizer_state
    # SMAC records the initial design in its embedded scenario.json, which a
    # resume honours (see `_initial_points`).
    scenario = optimizer_state.get(f"{subdir}/scenario.json") or {}
    pinned = (scenario.get("_meta") or {}).get("initial_design")
    if isinstance(pinned, dict) and pinned.get("n_configs") is not None:
        result.metadata["initial_design"] = pinned
    return result
