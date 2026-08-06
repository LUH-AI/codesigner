"""Running experiments: the create-and-run helper, and the background engine.

`run_experiment` runs synchronously and returns a result (used where a caller
wants to block). The background engine — `create_run` / `execute_run` /
`start_background_run`, with `DbCancelFlag` — records a run in the database, runs
it out of band rebuilding everything from the stored experiment, and writes the
result and status back so the page can poll and cancel.
"""

import random
import time
from pathlib import Path

from django.utils import timezone

from core import io
from core.io import _load_splits

from .. import permissions, registry
from ..registry import METRICS, MODELS, OPTIMIZERS
from . import snapshot as snapshot_adapter
from .run_logic import apply_metrics


def resolve_seed(seed_input: int) -> int:
    """A negative seed means 'pick one at random'; otherwise use it as given."""
    return random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)


def run_experiment(*, model_name, optimizer_name, dataset_path, seed,
                   primary_metric, n_trials, optimizer_params=None):
    """Build the split and run the chosen optimizer over all metrics.

    Every metric in the registry is scored on each trial; *primary_metric* is
    the one the optimizer optimizes. Returns the OptimizationResult.
    """
    model = MODELS[model_name]
    optimizer = type(OPTIMIZERS[optimizer_name])(**(optimizer_params or {}))
    X_train, X_val, y_train, y_val = _load_splits(Path(dataset_path), seed)
    return optimizer.optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric=primary_metric,
        n_trials=n_trials, previous_result=None, seed=seed, cancel_event=None,
    )


# ── Background run engine ──────────────────────────────────────────────────────

class DbCancelFlag:
    """A cancel flag backed by the Run row, honouring the optimizer's
    ``.is_set()`` contract. The value is cached for *ttl* seconds so the
    optimizer's per-trial checks don't hammer the database."""

    def __init__(self, run_id, ttl: float = 1.0):
        self.run_id = run_id
        self._ttl = ttl
        self._value = False
        self._checked = None

    def is_set(self) -> bool:
        from ..models import Run
        now = time.monotonic()
        if self._checked is None or (now - self._checked) >= self._ttl:
            self._value = Run.objects.filter(pk=self.run_id, cancel_requested=True).exists()
            self._checked = now
        return self._value


def create_run(experiment, n_trials, optimize_metric, started_by=None, stopping=None):
    """Record a pending run and commit its metric onto the experiment.

    Primary becomes the optimized metric; original is pinned on the first run.
    Returns the pending Run.
    """
    from ..models import Run

    primary, original = apply_metrics(experiment.original_metric, optimize_metric)
    experiment.primary_metric = primary
    experiment.original_metric = original
    experiment.save(update_fields=["primary_metric", "original_metric"])

    return Run.objects.create(
        experiment=experiment,
        n_trials=n_trials,
        primary_metric=optimize_metric,
        status="pending",
        started_by=started_by,
        stopping=stopping or {},
    )


def execute_run(run_id):
    """Run one optimization, synchronously, writing status and result to the DB.

    Rebuilds the experiment from its stored snapshot (so the worker needs only a
    row id), resumes from any prior result, and honours the DB cancel flag. On
    success the (possibly partial) result is saved and the run is marked done —
    or cancelled if cancellation was requested. Build/run failures mark the run
    errored without crashing the caller.
    """
    from ..models import Run

    run = Run.objects.get(pk=run_id)
    Run.objects.filter(pk=run_id).update(status="running", started_at=timezone.now())
    experiment = run.experiment

    try:
        # A custom model with a prepared environment runs in its own process, so
        # everything around it is built here but the model itself is not
        # imported. `launch` is None for a registry model, or for a custom one on
        # an instance without uv, and then it is imported as it always was.
        # Whose code is about to be executed. Checked here and not only at
        # the form because this is the process that would run it, and it is
        # reached by a task rather than by a request — a gate on the upload page
        # is advice, this is the decision.
        refusal = permissions.policy().custom_model_refusal(
            experiment, run.started_by or experiment.owner)
        if refusal:
            raise RuntimeError(refusal)

        launch, refusal = _model_launch(experiment)
        if refusal:
            raise RuntimeError(refusal)

        snapshot = snapshot_adapter.snapshot_from_experiment(experiment)
        _, built = io.build_experiment(
            snapshot, registry.METRICS, registry.MODELS, registry.OPTIMIZERS,
            read_only=False, load_model=launch is None,
        )
        optimizer = built["optimizer"]
        offset = len(built["result"].trials) if built["result"] else 0
        cancel = DbCancelFlag(run_id)

        def _optimize(model):
            return optimizer.optimize(
                model,
                built["X_train"], built["y_train"], built["X_val"], built["y_val"],
                metrics=built["metrics"],
                primary_metric=run.primary_metric,
                n_trials=run.n_trials,
                previous_result=built["result"],
                seed=built["seed"],
                cancel_event=cancel,
                stopping=run.stopping,
                splits=built["splits"],
            )

        if launch is None:
            result = _optimize(built["model"])
        else:
            from core.modelhost import model_session

            from . import modelenv

            with model_session(
                launch, built["splits"],
                seed=built["seed"], cancel=cancel, **modelenv.session_kwargs(),
            ) as model:
                result = _optimize(model)
    except Exception as exc:  # noqa: BLE001 — any failure is reported on the run
        Run.objects.filter(pk=run_id).update(
            status="error", error=str(exc), finished_at=timezone.now(),
        )
        return

    from ..models import Experiment

    cancelled = Run.objects.filter(pk=run_id, cancel_requested=True).exists()
    if result.trials:  # keep completed trials (progress survives a cancel)
        # filtered update, not .save(): a no-op if the experiment was deleted
        # mid-run (so a cancelled+deleted experiment is never resurrected).
        Experiment.objects.filter(pk=experiment.pk).update(
            result=optimizer.serialize_result(result),
        )
    # The trials this run added (excludes any resumed-from trials): their count
    # and total time, so the run box can show trials done, trial time, and the
    # search/bookkeeping overhead.
    new_trials = result.trials[offset:]
    Run.objects.filter(pk=run_id).update(
        status="cancelled" if cancelled else "done", finished_at=timezone.now(),
        trial_seconds=sum(t.duration for t in new_trials),
        trial_count=len(new_trials),
        # Empty when cancelled: the run did not end on its own terms.
        stopped_by="" if cancelled else (result.metadata.get("stopped_by") or ""),
    )


def _model_launch(experiment):
    """How to start this experiment's model: ``(launch_argv, refusal)``.

    ``(None, "")`` means import it in this process — a registry model, or a
    custom one on an instance that has no uv and no accounts. A refusal string
    means the run cannot go ahead and says why.
    """
    from . import modelenv

    if not experiment.model_file:
        return None, ""
    return modelenv.resolve_runner(experiment)


def start_background_run(run_id):
    """Enqueue a run for the huey consumer to execute (fire-and-forget).

    A lazy import keeps the task module out of the import cycle (tasks.py imports
    this module). A run whose thread dies with the server is swept back to
    `error` at startup. See services/dispatch.py for why immediate mode needs a
    thread.
    """
    from .. import tasks
    from .dispatch import enqueue

    enqueue(tasks.run_experiment_task, run_id)


def sweep_stale_runs():
    """Mark runs left ``running`` by an interrupted consumer as errored.

    With the durable huey queue a ``pending`` run is still queued and will be
    picked up, so it is not stale — only a ``running`` run is (its consumer died
    mid-execution). Run once at consumer/system startup. Finished runs untouched.
    """
    from ..models import Run

    return Run.objects.filter(status="running").update(
        status="error", error="Interrupted by a restart.",
    )
