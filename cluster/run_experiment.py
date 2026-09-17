"""Run one experiment, here, from files in a directory.

The other half of `ui/services/run.py`'s `execute_run`, for a machine that has
no database and no Django — a Slurm job on a cluster. A plain script rather than
a management command for exactly that reason: `manage.py` would need settings, a
database it never queries, and the whole web stack installed beside it.

That is affordable because `core/` imports no Django at all. What has to be
installed here is `cluster/requirements-cluster.txt` — six packages — and what
has to be copied here is `core/` and `model_sdk/`.

**The contract is a directory**, which is what crosses the machine boundary:

    snapshot.json   in    the experiment, in the .ihpo format it already has
    dataset.csv     in    what the snapshot's trials were measured on
    run.json        in    what bounds this run: criteria, metric, timeout
    CANCEL          in    created by the submitter; asks for a clean stop
    partial.json    out   the result so far, rewritten as trials land
    result.json     out   the finished result, in `Experiment.result` shape
    status.json     out   how it ended, and what the Run row needs to say so

`execute_run` reaches the database at three moments — a progress callback, a
cancellation check, and the final write. None of them is reachable from here, so
each has a file standing in for it; the optimizer itself is called exactly as it
is at home, and does not know the difference.
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
from core.optimizers.base import OptimizationResult      # noqa: E402
from core.registry import METRICS, MODELS, OPTIMIZERS    # noqa: E402

#: How often the result so far is rewritten. The submitter reads it to keep the
#: page's figures moving, so this is the resolution of "live"; it is also a whole
#: re-serialization of every trial, which is why it is not per trial. Mirrors
#: `PARTIAL_RESULT_SECONDS` on the application side.
PARTIAL_SECONDS = 5.0

#: Checked no more often than this, for the same reason `DbCancelFlag` caches:
#: the optimizer asks once per trial and an NFS stat per trial is waste.
CANCEL_TTL = 2.0


def write_json(path: Path, payload) -> None:
    """Write *payload* so a reader never sees half of it.

    The submitter is polling these files across NFS while they are being
    written. `os.replace` is atomic within a directory, so a reader sees either
    the previous complete file or the next one — never a truncated parse error
    that would look like a corrupted run.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(io.to_bytes(payload).decode("utf-8"), encoding="utf-8")
    os.replace(tmp, path)


class FileCancelFlag:
    """`cancel_event` backed by a file the submitter creates.

    The optimizer's contract is `.is_set()`, the same one `DbCancelFlag`
    answers. Asking the filesystem is how a machine with no database learns that
    somebody pressed Cancel — and it is a *request*, not a kill: the run breaks
    out of its loop and still writes what it has, which is what keeps the trials
    already paid for. `scancel` is the other thing, and it is not this.
    """

    def __init__(self, path: Path, ttl: float = CANCEL_TTL):
        self._path = path
        self._ttl = ttl
        self._value = False
        self._checked = None

    def is_set(self) -> bool:
        now = time.monotonic()
        if self._checked is None or (now - self._checked) >= self._ttl:
            self._value = self._path.exists()
            self._checked = now
        return self._value


def partial_writer(workdir: Path, optimizer, previous_result, primary_metric):
    """A `BaseOptimizer.progress` callback that writes the trials so far.

    The same shape as the application's `_partial_result_writer`, and empty in
    the same places: importance, interactions, PDP and local effects each cost a
    surrogate fit or 2^n_hp coalitions and none may run per trial. A partial
    result leaves those fields empty, which is the state a cancelled run already
    produces and the page already handles.
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

    # The snapshot names the dataset by the path it had on the machine that
    # wrote it, which does not exist here. Repointed at what was staged beside
    # it — the same rewrite the export view performs on the way out.
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

    cancel = FileCancelFlag(workdir / "CANCEL")
    result = optimizer.optimize(
        built["model"],
        built["X_train"], built["y_train"], built["X_val"], built["y_val"],
        metrics=built["metrics"],
        primary_metric=primary_metric,
        previous_result=previous,
        seed=built["seed"],
        cancel_event=cancel,
        stopping=settings["stopping"],
        splits=built["splits"],
    )

    write_json(workdir / "result.json", optimizer.serialize_result(result))
    write_json(workdir / "status.json", {
        "state": "cancelled" if cancel.is_set() else "done",
        "stopped_by": result.metadata.get("stopped_by") or "",
        "offset": offset,
        # Captured here for the same reason the application captures it at the
        # start of a run: this is the only moment there is a model to ask, and
        # every surrogate-backed figure needs the space afterwards.
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path,
                        help="directory holding snapshot.json, dataset.csv and run.json")
    workdir = parser.parse_args(argv).workdir

    try:
        return run(workdir)
    except Exception as exc:  # noqa: BLE001 — the submitter reads the reason, not a traceback
        # Written rather than raised: the submitter is polling files, and a job
        # that dies leaving no status is indistinguishable from one still going.
        write_json(workdir / "status.json", {
            "state": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stopped_by": "", "offset": 0, "config_space": None,
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
