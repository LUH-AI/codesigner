"""The script runs one experiment from the files in a directory.

A Slurm node runs the script without Django or a database. The directory holds
the inputs `snapshot.json`, `dataset.csv`, `run.json` and an optional
`model.py`, and the script writes `partial.json`, `result.json` and
`status.json` there. The submitter creates a `CANCEL` file to ask for a clean
stop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import io                                      # noqa: E402
from core.optimizers.base import OptimizationResult, PolledFlag  # noqa: E402
from core.optimizers.trial import scoreable                # noqa: E402
from core.registry import METRICS, MODELS, OPTIMIZERS    # noqa: E402

#: Each rewrite of `partial.json` re-serializes every trial;
#: `PARTIAL_RESULT_SECONDS` is the app-side twin.
PARTIAL_SECONDS = 5.0

#: `CANCEL_TTL` spares an NFS stat on every trial.
CANCEL_TTL = 2.0


def write_json(path: Path, payload) -> None:
    """`os.replace` is atomic within a directory, so the submitter polling over
    NFS never reads a truncated file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(io.to_bytes(payload).decode("utf-8"), encoding="utf-8")
    os.replace(tmp, path)


def partial_writer(workdir: Path, optimizer, previous_result, primary_metric):
    """Like `_partial_result_writer`, the callback leaves the costly analytics
    fields empty.
    """
    previous_trials = previous_result.trials if previous_result else []
    state = {"at": time.monotonic()}

    def write(collector) -> bool:
        now = time.monotonic()
        if now - state["at"] < PARTIAL_SECONDS:
            return False
        state["at"] = now
        partial = OptimizationResult(
            trials=previous_trials + collector.results,
            primary_metric=primary_metric,
            best_config=collector.incumbent_config or {},
            best_score=collector.incumbent_score,
            hyperparameter_importance={},
            hyperparameter_importance_warning={},
        )
        write_json(workdir / "partial.json", optimizer.serialize_result(partial))
        return True

    return write


def run(workdir: Path) -> int:
    snapshot = json.loads((workdir / "snapshot.json").read_text(encoding="utf-8"))
    settings = json.loads((workdir / "run.json").read_text(encoding="utf-8"))

    # The original dataset path exists only on the machine that wrote the
    # snapshot.
    snapshot["dataset"]["path"] = str(workdir / "dataset.csv")

    model_file = workdir / "model.py"
    if model_file.exists():
        snapshot["model"]["path"] = str(model_file)

    _, built = io.build_experiment(snapshot, METRICS, MODELS, OPTIMIZERS,
                                   read_only=False, load_model=True)
    optimizer = built["optimizer"]
    previous = built["result"]
    offset = len(previous.trials) if previous else 0
    primary_metric = settings["primary_metric"]

    optimizer.analytics_max_coalitions = settings.get("analytics_max_coalitions") or None
    optimizer.analytics_wanted = settings.get("analytics_wanted", True)
    optimizer.progress = partial_writer(workdir, optimizer, previous, primary_metric)

    # `CANCEL` asks for a clean stop that keeps finished trials, unlike
    # `scancel`.
    cancel = PolledFlag((workdir / "CANCEL").exists, ttl=CANCEL_TTL)
    result = optimizer.optimize(
        built["model"],
        built["X_train"], built["y_train"], built["X_val"], built["y_val"],
        metrics=scoreable(built["metrics"], built["model"], primary_metric),
        primary_metric=primary_metric,
        previous_result=previous,
        seed=built["seed"],
        cancel_event=cancel,
        stopping=settings["stopping"],
        priors=settings.get("priors"),
        configurations=settings.get("configurations"),
        splits=built["splits"],
    )

    write_json(workdir / "result.json", optimizer.serialize_result(result))
    write_json(workdir / "status.json", {
        "state": "cancelled" if cancel.is_set() else "done",
        "stopped_by": result.metadata.get("stopped_by") or "",
        "offset": offset,
        # `prior` records whether the prior was applied, skipped or refused, and
        # at which trial.
        "prior": result.metadata.get("prior"),
        # `_config_space` asks the model now because surrogate-backed figures
        # need the space after the run.
        "config_space": _config_space(built),
        "error": "",
    })
    return 0


def _config_space(built):
    model = built.get("model")
    if model is None:
        return None
    try:
        return model.get_config_space(seed=built["seed"]).to_serialized_dict()
    except Exception:  # noqa: BLE001 — bookkeeping must not fail a finished run
        return None


def main(argv=None) -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path,
                        help="directory holding snapshot.json, dataset.csv and run.json")
    workdir = parser.parse_args(argv).workdir

    try:
        return run(workdir)
    except Exception as exc:  # noqa: BLE001 — the submitter reads the reason, not a traceback
        # A job that dies without `status.json` looks like one still running.
        write_json(workdir / "status.json", {
            "state": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stopped_by": "", "offset": 0, "config_space": None,
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
