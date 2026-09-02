"""Reading a SMAC output directory as an experiment.

SMAC writes five files — `runhistory.json`, `configspace.json`, `scenario.json`,
`intensifier.json`, `optimization.json` — into `<output>/<name>/<seed>/`. This
turns them into a snapshot the rest of the application already understands, and
that is the whole of the integration: there is no second internal
representation, and everything downstream of `io.parse` is untouched.

It slots in as cleanly as it does because `.ihpo`'s `result` block *is*
`runhistory.json`: the same four top-level keys, and each data entry carrying
SMAC's own ten fields verbatim (`RUN_INFO_KEYS`) plus three of ours — `scores`,
`incumbent_score`, `incumbent_config_id`. So most of this module is not
translation. It is the handful of places where the two genuinely differ:

- **A configuration can be evaluated more than once.** SMAC keys its data by
  `(config_id, instance, seed, budget)`, so one config id can head several rows.
  `.ihpo` holds `config_id == trial number`, an invariant the whole selection
  layer addresses trials through, so rows are renumbered sequentially and the
  key they came from is kept in `additional_info`.
- **Its cost is not our score.** SMAC minimises an objective it never names a
  direction for, because it has only one. That objective is imported as what it
  is — a lower-is-better, unbounded metric the file declares for itself.
- **It writes `Infinity` and `NaN`.** Both are bare JSON tokens that Python
  reads and stricter parsers do not, and neither is a measurement.
- **It records trials that had not finished.** A RUNNING row carries a
  placeholder cost, which is a trial that never happened reported as one that
  scored badly.

What does not come across is anything about *what was being optimized*: a
`Scenario` records no dataset, no target function, not even a name for one. So
an imported run is read-only, and cannot be made otherwise by attaching a
dataset here — nothing in the files could check that it was the right one.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .io import SNAPSHOT_FORMAT
from .metrics import LABELS, Metric, describe
from .optimizers.timing import RUN_INFO_KEYS, STATUS_SUCCESS

#: The files SMAC writes, by basename. Only the first three are read; the other
#: two are carried through so that a re-export of an imported run is still a
#: complete copy of what was imported.
RUNHISTORY = "runhistory.json"
CONFIGSPACE = "configspace.json"
SCENARIO = "scenario.json"
CARRIED = ("intensifier.json", "optimization.json")

#: SMAC's `StatusType.RUNNING`. Its rows are bookkeeping for a trial that had
#: been asked for and not yet answered, and carry a placeholder cost.
STATUS_RUNNING = 0

#: What an imported objective is called. Namespaced, and deliberately so: a
#: metric's declaration is written into a file only for names the registry does
#: not have (see `BaseOptimizer.serialize_result`), so an objective that happened
#: to be called `accuracy` would lose its declaration on the first re-export and
#: read back as a 0-to-1 higher-is-better score. A prefix makes that collision
#: impossible without changing what gets serialized for everybody else.
NAMESPACE = "smac:"


class SmacImportError(ValueError):
    """A directory that cannot be read as one run, with the reason."""


def snapshot_from_smac(files: Mapping[str, Any], *, name: str = "") -> dict:
    """A parsed SMAC output directory as a format-2 snapshot dict.

    *files* maps each file's path — relative to whatever was uploaded — to its
    parsed JSON. Extra files are ignored; the three that matter are found by
    basename, so it does not matter whether the upload was the run directory
    itself or the `smac3_output` above it.

    *name* names the experiment; the scenario's own run name is used when it is
    left empty.

    Raises `SmacImportError`, with a reason worth showing a reader, when the
    directory is not one readable single-objective run.
    """
    found = _locate(files)
    runhistory = found[RUNHISTORY]
    scenario = found.get(SCENARIO) or {}

    objective = _objective(scenario)
    metric_name = NAMESPACE + objective
    metric = Metric(name=metric_name, fn=None, higher_is_better=False,
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
        # The application version is the one reading it. The file this came from
        # was not written by codesigner and has no such field.
        "version": _version(),
        "name": name or str(scenario.get("name") or "Imported SMAC run"),
        "seed": seed,
        "dataset": {"filename": "", "path": ""},
        "model": {"kind": "external", "name": _model_name(scenario), "path": ""},
        # Not recorded anywhere in a SMAC output. Left at the shape the rest of
        # the application expects rather than guessed at; the page says the
        # scheme is not known rather than reporting this 0.2 as a fact.
        "evaluation": {"scheme": "holdout", "folds": None, "test_size": None,
                       "stratified": None},
        "metrics": {"names": [metric_name], "current": metric_name,
                    "original": metric_name},
        "optimizer": {"name": "SMAC", "params": {}},
        "space": found[CONFIGSPACE],
        "result": {
            "stats": {"submitted": len(trials), "finished": len(trials),
                      "running": 0},
            "data": trials,
            "configs": configs,
            "config_origins": origins,
            # Every file carried verbatim, so a re-export of this experiment is
            # still a complete copy of the directory it came from.
            "optimizer_state": _carried(files),
            "primary_metric": metric_name,
            "declared_metrics": {metric_name: describe(metric)},
            "best_score": min(t["scores"][metric_name] for t in trials),
            "best_config_id": str(_best(trials, metric_name)),
            "trials_limit": _int(scenario.get("n_trials"), 0) or None,
        },
    }


# ── finding the files ────────────────────────────────────────────────────────

def _locate(files: Mapping[str, Any]) -> dict:
    """The three files that are read, by basename, or a reason there are not.

    By basename because a browser's directory picker reports paths relative to
    whichever folder was chosen, and both the run directory and the
    `smac3_output` above it are reasonable things to choose.
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
    """Every SMAC file, keyed by a path safe to store.

    Flattened to basenames rather than kept at their uploaded depth: the keys
    are validated as relative paths on the way into an `.ihpo`
    (`io._check_optimizer_state`), and a browser reports whatever the person
    picked, which is not something to write into a file unexamined.
    """
    wanted = (RUNHISTORY, CONFIGSPACE, SCENARIO) + CARRIED
    carried = {}
    for path, content in files.items():
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if base in wanted and base not in carried:
            carried[base] = content
    return carried


# ── the trials ───────────────────────────────────────────────────────────────

def _trials(runhistory: Mapping[str, Any], metric_name: str) -> list:
    """`runhistory["data"]`, renumbered and cleaned, as `.ihpo` data entries.

    Each kept row carries SMAC's own fields through unchanged and gains the
    three `.ihpo` adds. Two kinds of row are dropped:

    - **RUNNING**, whose cost is a placeholder for an answer that never came.
    - anything whose cost is not a finite number. SMAC records `crash_cost` for
      a trial that failed, which is `inf` unless the run set otherwise, and
      there is no honest score to give it here: the importer has no dataset, so
      it cannot compute what a model that knew nothing would have scored.

    A crashed trial whose recorded cost *is* finite is kept, marked failed. That
    is a real measurement of a configuration that does not work, which is a
    result the search acted on.
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
        entry = {key: _sanitised(row.get(key)) for key in RUN_INFO_KEYS
                 if key in row}
        entry["status"] = status
        # Where this row came from, so a renumbered trial can still be traced
        # back to the runhistory it was read out of.
        entry["additional_info"] = {
            **(entry.get("additional_info") or {}),
            "smac_key": {"config_id": _int(row.get("config_id"), 0),
                         "instance": row.get("instance"),
                         "seed": row.get("seed"),
                         "budget": _finite(row.get("budget"))},
        }
        entry["config_id"] = number
        entry["cost"] = cost
        # A lower-is-better metric stores itself unchanged, so the cost and the
        # score are the same number — see `core.metrics.to_cost`. Written out
        # rather than left implied, because every figure reads `scores`.
        entry["scores"] = {metric_name: cost}
        entry["_config"] = dict(config)
        entry["_origin"] = str(origins.get(config_id) or "")
        trials.append(entry)
    return trials


def _record_incumbents(trials: list, metric_name: str) -> None:
    """Fill in each trial's running best, in place.

    SMAC does not store a trajectory — it recomputes one from the runhistory
    when it wants one, and so does every figure here (`incumbent_scores`). These
    two fields are provenance: what was the best when this trial ran.
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
    """The cheapest trial's number. Ties go to the earliest, as `min` does."""
    return min(trials, key=lambda t: t["scores"][metric_name])["config_id"]


# ── reading the scenario ─────────────────────────────────────────────────────

def _objective(scenario: Mapping[str, Any]) -> str:
    """The single objective this run minimised, or a refusal naming why not.

    Multi-objective is refused rather than reduced to its first component: the
    numbers are a trade-off surface, and picking one of them silently would
    report a search for something nobody ran.
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
    """What to call the thing that was optimized.

    A `Scenario` records nothing about it — not a name, not a path, not a
    digest — so this says where the run came from instead of inventing a model
    it cannot know. See this module's docstring on why that is also the reason
    an imported run cannot be resumed.
    """
    run = str(scenario.get("name") or "").strip()
    return f"SMAC run ({run})" if run else "SMAC run"


def _version() -> str:
    from importlib.metadata import version

    try:
        return version("codesigner")
    except Exception:  # noqa: BLE001 — a version is a record, not a requirement
        return "0"


# ── numbers that a strict reader can carry ───────────────────────────────────

def _finite(value) -> Optional[float]:
    """*value* as a float, or None when it is not a number a file can hold.

    `json.loads` reads SMAC's bare `Infinity`/`NaN` tokens happily and produces
    floats that `json.dumps` writes straight back out — which is how an `.ihpo`
    ends up unreadable by anything stricter than Python.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _sanitised(value):
    """*value* with any non-finite number in it replaced by None, recursively."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitised(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitised(v) for v in value]
    return value


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
