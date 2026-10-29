"""`snapshot_from_smac` turns SMAC's five output files into a snapshot, and
everything downstream of `io.parse` stays untouched. The `.ihpo` `result` block
mirrors `runhistory.json`, so the module only handles the differences: repeated
configurations, an unnamed minimized cost, bare `Infinity` and `NaN` tokens, and
unfinished RUNNING rows.

A `Scenario` records nothing about what was optimized, so an imported run stays
read-only, and attaching a dataset cannot make it runnable.
"""

from __future__ import annotations

import math
from importlib.metadata import version as dist_version
from typing import Any, Mapping, Optional

from ...io import SNAPSHOT_FORMAT
from ...jsonable import finite
from ...metrics import LABELS, describe, Metric, unscoreable
from ..timing import RUN_INFO_KEYS, STATUS_SUCCESS
from .params import INITIAL_DESIGNS, STRATEGIES

#: Only the first three are read; the other two are carried for a complete
#: re-export.
RUNHISTORY = "runhistory.json"
CONFIGSPACE = "configspace.json"
SCENARIO = "scenario.json"
CARRIED = ("intensifier.json", "optimization.json")

#: SMAC's `StatusType.RUNNING`; its rows carry a placeholder cost.
STATUS_RUNNING = 0

#: A metric declaration is only serialized for names missing from the registry,
#: so the prefix keeps an objective called `accuracy` from reading back as a
#: 0-to-1 score.
NAMESPACE = "smac:"


class SmacImportError(ValueError):
    """`SmacImportError` explains why a directory is unreadable."""


def snapshot_from_smac(files: Mapping[str, Any], *, name: str = "") -> dict:
    """`_locate` finds the needed files in *files* by basename, wherever the
    upload put them. An empty *name* falls back to the scenario's run name, and
    an unreadable or multi-objective run raises `SmacImportError`.
    """
    found = _locate(files)
    runhistory = found[RUNHISTORY]
    scenario = found.get(SCENARIO) or {}

    objective = _objective(scenario)
    metric_name = NAMESPACE + objective
    metric = Metric(name=metric_name, fn=unscoreable, higher_is_better=False,
                    bounds=(None, None), needs=LABELS)

    trials = _trials(runhistory, metric_name)
    if not trials:
        raise SmacImportError(
            "this run has no finished trials to import — every entry in its "
            "runhistory is still running, or carries no usable cost")

    configs = {str(t["config_id"]): t.pop("_config") for t in trials}
    origins = {str(t["config_id"]): t.pop("_origin") for t in trials}
    _record_incumbents(trials, metric_name)

    seed = _int(scenario.get("seed"), 0)
    return {
        "format": SNAPSHOT_FORMAT,
        # Version of the reading application; SMAC files have none.
        "version": dist_version("codesigner"),
        "name": name or str(scenario.get("name") or "Imported SMAC run"),
        "seed": seed,
        "dataset": {"filename": "", "path": ""},
        "model": {"kind": "external", "name": _model_name(scenario), "path": ""},
        # SMAC records no evaluation, so the evaluation fields stay null and the
        # page reports the scheme as unrecorded.
        "evaluation": {"scheme": "holdout", "folds": None, "test_size": None,
                       "stratified": None},
        "metrics": {"names": [metric_name], "current": metric_name,
                    "original": metric_name},
        "optimizer": {"name": "SMAC", "params": _optimizer_settings(scenario)},
        "space": found[CONFIGSPACE],
        "result": {
            "stats": {"submitted": len(trials), "finished": len(trials),
                      "running": 0},
            "data": trials,
            "configs": configs,
            "config_origins": origins,
            "optimizer_state": _carried(files),
            "primary_metric": metric_name,
            "declared_metrics": {metric_name: describe(metric)},
            "best_score": min(t["scores"][metric_name] for t in trials),
            "best_config_id": str(_best(trials, metric_name)),
            "trials_limit": _int(scenario.get("n_trials"), 0) or None,
        },
    }


def _locate(files: Mapping[str, Any]) -> dict:
    """`_locate` matches basenames because a directory picker reports paths
    relative to the chosen folder.
    """
    found: dict[str, Any] = {}
    for path, content in files.items():
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if base in (RUNHISTORY, CONFIGSPACE, SCENARIO) and base not in found:
            found[base] = content

    for required in (RUNHISTORY, CONFIGSPACE):
        if required not in found:
            raise SmacImportError(f"no {required} in this directory")
    if not isinstance(found[RUNHISTORY].get("data"), list):
        raise SmacImportError(f"{RUNHISTORY} has no trial data")
    return found


def _carried(files: Mapping[str, Any]) -> dict:
    """Keys are flattened to basenames because `io.check_optimizer_state`
    validates them as relative paths.
    """
    wanted = (RUNHISTORY, CONFIGSPACE, SCENARIO) + CARRIED
    carried = {}
    for path, content in files.items():
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if base in wanted and base not in carried:
            carried[base] = content
    return carried


def _trials(runhistory: Mapping[str, Any], metric_name: str) -> list:
    """`_trials` renumbers rows and drops RUNNING rows and rows without a finite
    cost, since the importer has no dataset to score a crash. A crashed trial
    with a finite cost stays, marked failed.
    """
    configs = runhistory.get("configs") or {}
    origins = runhistory.get("config_origins") or {}

    trials = []
    for row in runhistory["data"]:
        if not isinstance(row, dict):
            continue
        status = _int(row.get("status"), STATUS_SUCCESS)
        if status == STATUS_RUNNING:
            continue
        cost = _finite(row.get("cost"))
        if cost is None:
            continue
        config_id = str(row.get("config_id"))
        config = configs.get(config_id)
        if not isinstance(config, dict):
            continue

        number = len(trials) + 1
        entry = {key: finite(row.get(key)) for key in RUN_INFO_KEYS
                 if key in row}
        entry["status"] = status
        # Traces a renumbered trial back to its runhistory row.
        entry["additional_info"] = {
            **(entry.get("additional_info") or {}),
            "smac_key": {"config_id": _int(row.get("config_id"), 0),
                         "instance": row.get("instance"),
                         "seed": row.get("seed"),
                         "budget": _finite(row.get("budget"))},
        }
        entry["config_id"] = number
        entry["cost"] = cost
        # Cost equals score for a lower-is-better metric; every figure reads
        # `scores`.
        entry["scores"] = {metric_name: cost}
        entry["_config"] = dict(config)
        entry["_origin"] = str(origins.get(config_id) or "")
        trials.append(entry)
    return trials


#: `_FACADE_STRATEGIES` and `_DESIGNS` map SMAC's names, recorded in
#: `scenario.json`'s `_meta`, to codesigner settings. The importer skips
#: `local_search_iterations` and `max_features`, which have no unambiguous
#: match.
_FACADE_STRATEGIES = {facade.__name__: key for key, facade in STRATEGIES.items()}
_DESIGNS = {design.__name__: key for key, design in INITIAL_DESIGNS.items()}
#: SMAC's "no limit" depth, skipped on import.
_UNBOUNDED_DEPTH = 2 ** 20


def _optimizer_settings(scenario: dict) -> dict:
    """`_optimizer_settings` reads the run's settings from its scenario `_meta`,
    so a re-run uses them instead of codesigner's defaults. `known_params` drops
    settings this version lacks.
    """
    from .optimizer import SMACOptimizer

    meta = scenario.get("_meta") or {}
    if not meta:
        return {}

    def block(name: str) -> Any:
        return meta.get(name) or {}

    out: dict = {}
    strategy = _FACADE_STRATEGIES.get(block("facade").get("name"))
    if strategy:
        out["search_strategy"] = strategy

    acquisition = block("acquisition_function")
    if acquisition.get("name"):
        out["acquisition"] = str(acquisition["name"]).lower()
    if acquisition.get("xi") is not None:
        out["acquisition_xi"] = float(acquisition["xi"])

    challengers = block("acquisition_maximizer").get("challengers")
    if challengers is not None:
        out["challengers"] = int(challengers)

    design = _DESIGNS.get(block("initial_design").get("name"))
    if design:
        out["initial_design"] = design

    probability = block("random_design").get("probability")
    if probability is not None:
        out["random_probability"] = float(probability)

    retrain = block("config_selector").get("retrain_after")
    if retrain is not None:
        out["retrain_after"] = int(retrain)

    if scenario.get("use_default_config") is not None:
        out["use_default_config"] = bool(scenario["use_default_config"])

    model = block("model")
    if strategy == "rf":
        for setting, recorded in (("rf_trees", "n_estimators"),
                                  ("rf_min_samples_split", "min_samples_split"),
                                  ("rf_min_samples_leaf", "min_samples_leaf")):
            if model.get(recorded) is not None:
                out[setting] = int(model[recorded])
        depth = model.get("max_depth")
        if depth is not None and int(depth) < _UNBOUNDED_DEPTH:
            out["rf_max_depth"] = int(depth)
        if model.get("bootstrap") is not None:
            out["rf_bootstrapping"] = bool(model["bootstrap"])
    elif strategy == "gp":
        # The GP facades differ in how kernel hyperparameters are fitted.
        out["gp_model_type"] = ("mcmc" if "MCMC" in str(model.get("name") or "")
                                else "vanilla")

    return SMACOptimizer.known_params(out)


def _record_incumbents(trials: list, metric_name: str) -> None:
    """SMAC stores no trajectory, so the incumbent fields record the best score
    at each trial's time.
    """
    best_score: Optional[float] = None
    best_id = 1
    for entry in trials:
        score = entry["scores"][metric_name]
        if best_score is None or score < best_score:
            best_score, best_id = score, entry["config_id"]
        entry["incumbent_score"] = best_score
        entry["incumbent_config_id"] = best_id


def _best(trials: list, metric_name: str) -> int:
    return min(trials, key=lambda t: t["scores"][metric_name])["config_id"]


def _objective(scenario: Mapping[str, Any]) -> str:
    """`_objective` only accepts a single objective and raises a
    `SmacImportError` for a multi-objective run, since picking one component
    would report a search nobody ran.
    """
    objectives = scenario.get("objectives", "cost")
    if isinstance(objectives, (list, tuple)):
        if len(objectives) != 1:
            raise SmacImportError(
                "this run optimized several objectives at once "
                f"({', '.join(str(o) for o in objectives)}), and an experiment "
                "here has one — import it per objective, or not at all")
        objectives = objectives[0]
    return str(objectives or "cost")


def _model_name(scenario: Mapping[str, Any]) -> str:
    """`_model_name` points at the source run because a `Scenario` records no
    model.
    """
    run = str(scenario.get("name") or "").strip()
    return f"SMAC run ({run})" if run else "SMAC run"


def _finite(value) -> Optional[float]:
    """`json.loads` accepts SMAC's `Infinity` and `NaN`, and `json.dumps` would
    write them back into an `.ihpo` that stricter readers reject.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
