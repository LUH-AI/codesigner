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

from django.conf import settings

from .settings import eager_analytics_wanted, resolve_settings
from django.utils import timezone

from core import io
from core.io import DEFAULT_TEST_SIZE, _load_splits

from core.optimizers.base import STOPPED_BY_CANCELLED, OptimizationResult

from .. import permissions, registry
from ..registry import METRICS, MODELS, OPTIMIZERS
from . import snapshot as snapshot_adapter
from .run_logic import apply_metrics


def resolve_seed(seed_input: int) -> int:
    """A negative seed means 'pick one at random'; otherwise use it as given."""
    return random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)


def run_experiment(*, model_name, optimizer_name, dataset_path, seed,
                   primary_metric, n_trials, optimizer_params=None,
                   test_size=None):
    """Build the split and run the chosen optimizer over all metrics.

    Every metric in the registry is scored on each trial; *primary_metric* is
    the one the optimizer optimizes. Returns the OptimizationResult.
    """
    model = MODELS[model_name]
    optimizer = type(OPTIMIZERS[optimizer_name])(**(optimizer_params or {}))
    X_train, X_val, y_train, y_val = _load_splits(
        Path(dataset_path), seed, test_size or DEFAULT_TEST_SIZE)
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


def create_run(experiment, stopping, optimize_metric, started_by=None,
               trial_timeout=None):
    """Record a pending run and commit its metric onto the experiment.

    `current_metric` becomes the optimized metric; original is pinned on the
    first run. *stopping* is the criteria the run ends on — at least one; see
    `core.optimizers.base.STOPPING_CRITERIA`. *trial_timeout* is how long any
    one call to the model may take — see `core.modelhost.deadline` — which is a
    limit on a trial rather than on the run, and so is not one of them. Returns
    the pending Run.

    The optimizer's settings are copied onto the run rather than referenced.
    They are editable between runs, so the experiment's current settings are not
    the ones the earlier trials came out of, and a record that said they were
    would be wrong about every experiment anyone ever adjusted.
    """
    from ..models import Run

    events = _metric_change_event(experiment, optimize_metric)
    current, original = apply_metrics(experiment.original_metric, optimize_metric)
    experiment.current_metric = current
    experiment.original_metric = original
    experiment.save(update_fields=["current_metric", "original_metric"])

    return Run.objects.create(
        experiment=experiment,
        primary_metric=optimize_metric,
        status="pending",
        started_by=started_by,
        stopping=dict(stopping),
        trial_timeout=dict(trial_timeout or {}),
        events=events,
    )


def _metric_change_event(experiment, now):
    """The record of an experiment changing what it optimizes, if it just did.

    Worth recording because of what it costs rather than because it happened:
    the accumulated trials are re-read under the new metric, and an optimizer
    that carries a fitted model of the objective has to throw it away — it was
    fitted to costs from a different question. That is the one thing an .ihpo
    could not previously say about its own history.

    *was* comes from the stored result rather than `experiment.current_metric`,
    which is not the same thing: `create_run` commits `current_metric` onto the
    experiment whether or not the run it is starting ever produces a trial. A
    run that changes the metric and then errors leaves `current_metric` pointed
    at a metric no trial was ever scored under — a retry under that same metric
    would then see no difference and record no event, silently losing the one
    change that actually happened, while the failed run keeps a misleading one
    attached to zero trials. The stored result only moves on an actual trial,
    so reading `was` from it and *trials* from it are the same source agreeing
    with itself, not two clocks that can drift.
    """
    stored = experiment.result or {}
    trials = stored.get("data") or []
    was = stored.get("primary_metric")
    if not was or not trials or was == now:
        return []

    optimizer = registry.OPTIMIZERS.get(experiment.optimizer_name)
    return [{
        "kind": "metric_changed",
        "from": was,
        "to": now,
        "at_trial": len(trials),
        "surrogate": ("rebuilt_and_replayed"
                      if getattr(optimizer, "fits_surrogate", False) else "none"),
    }]


#: How often a run at most writes what it has so far. Throttled by wall clock
#: rather than by a trial count, because trial durations span three orders of
#: magnitude here: at one write per trial a run of 0.07s trials would rewrite the
#: whole result a dozen times a second, and a count that fixed that would starve
#: a run of ten-minute trials of updates for an hour. Serializing costs O(bytes
#: so far), so a time bound is what actually bounds the total — one write per
#: interval however fast the trials arrive.
PARTIAL_RESULT_SECONDS = 5.0


def _partial_result_writer(experiment_pk, optimizer, previous_result, primary_metric):
    """A `BaseOptimizer.progress` callback that saves the run's trials so far.

    **The constraint was never rendering, it was persistence.** Every
    trial-based figure — trial performance, trial duration, the trials table,
    parallel coordinates, the projection, best and selected configuration —
    needs nothing but the trials, and the page has always been able to draw
    them. It simply had nothing to draw: `execute_run` wrote
    `experiment.result` exactly once, after `_optimize` returned.

    What is deliberately *not* here is the analytics. Importance, interactions,
    PDP and local effects each cost a surrogate fit or 2^n_hp coalitions, and
    none of them may run per trial. A partial result leaves those fields empty,
    which is a state the page already handles — it is what a cancelled run
    produces (see `BaseOptimizer.compute_hp_games`).

    The channel is the database, not the process: under a real huey consumer the
    run is in a worker process and nothing in-process could hand the page
    anything. Written with the same filtered `update()` the final write uses, so
    an experiment deleted mid-run is not resurrected by its own run finishing a
    trial.
    """
    from ..models import Experiment

    previous_trials = previous_result.trials if previous_result else []
    state = {"at": time.monotonic()}

    def write(collector) -> bool:
        now = time.monotonic()
        if now - state["at"] < PARTIAL_RESULT_SECONDS:
            return False
        state["at"] = now

        trials = previous_trials + collector.results
        partial = OptimizationResult(
            trials=trials,
            primary_metric=primary_metric,
            best_config=collector.incumbent_config or {},
            best_score=collector.incumbent_score,
            # Empty, not absent: see above. Every `hyperparameter_*` field
            # defaults to empty, and the two that do not have defaults are
            # given them here.
            hyperparameter_importance={},
            hyperparameter_importance_warning={},
        )
        Experiment.objects.filter(pk=experiment_pk).update(
            result=optimizer.serialize_result(partial))
        return True

    return write


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
        # What this machine will spend on the analytics computed at run
        # completion. Set here rather than read inside `core/`, which stays
        # Django-free; see BaseOptimizer.eager_analytics_budget_exceeded for what
        # the number counts. 0 means no limit.
        optimizer.analytics_max_coalitions = (
            settings.ANALYTICS_EAGER_MAX_COALITIONS or None)
        # And whether anything will show them. With both figures that display
        # the games switched off, computing them fills fields no page reads —
        # 2^n_hp coalition evaluations per game per metric, for nobody. Read
        # here rather than in `core/`, same as the budget above.
        optimizer.analytics_wanted = eager_analytics_wanted(resolve_settings(run.experiment))
        # And how the page sees any of it before the run ends. Same reasoning
        # about where this is set: the callback writes to the database, which
        # `core/` knows nothing about.
        optimizer.progress = _partial_result_writer(
            experiment.pk, optimizer, built["result"], run.primary_metric)
        offset = len(built["result"].trials) if built["result"] else 0
        cancel = DbCancelFlag(run_id)

        def _optimize(model):
            return optimizer.optimize(
                model,
                built["X_train"], built["y_train"], built["X_val"], built["y_val"],
                metrics=built["metrics"],
                primary_metric=run.primary_metric,
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
                seed=built["seed"], cancel=cancel,
                **modelenv.session_kwargs(run.trial_timeout, seed=built["seed"]),
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
        # With the count, this is the range of trials this run produced.
        trial_offset=offset,
        # Being interrupted is a reason a run stopped, and the page has to be
        # able to say so. Left to the criteria only when it was not.
        stopped_by=(STOPPED_BY_CANCELLED if cancelled
                    else (result.metadata.get("stopped_by") or "")),
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


#: When this process started. In immediate mode it is also when every run this
#: process could possibly be executing started, which is what `sweep_orphaned_runs`
#: needs and cannot get any other way.
PROCESS_STARTED = timezone.now()

#: Once per process, and only because there is no startup hook that may touch the
#: database — see ui/apps.py for why `ready()` deliberately does not.
_SWEPT = False


def sweep_orphaned_runs():
    """Finish runs whose executor is provably gone. Returns how many.

    Only meaningful in **immediate mode**, where huey executes a task in the
    calling process rather than in a consumer — see services/dispatch.py, which
    puts it on a daemon thread so the request can return. A daemon thread dies
    with its process, and in development that process is `runserver`, which
    restarts every time a file is saved. So a run started before this process
    booted has no one executing it, and never will.

    That is an exact test rather than a timeout: in immediate mode the executor
    *is* this process, so "older than this process" means orphaned, with no
    guessing about how long a run ought to take. It is wrong in consumer mode,
    where the executor is a different process that outlives any web restart —
    which is why this checks. Consumer restarts are covered by the
    `sweep_stale_runs` management command at startup.

    Cancelling one of these does nothing, and that is the reported symptom:
    `cancel_requested` is a flag the *executing* thread polls, so with no thread
    there is nobody to read it and the run sits at "running" for ever.
    """
    global _SWEPT
    from huey.contrib.djhuey import HUEY

    from ..models import Run

    if _SWEPT or not HUEY.immediate:
        return 0
    _SWEPT = True
    return Run.objects.filter(status="running", started_at__lt=PROCESS_STARTED).update(
        status="error", finished_at=timezone.now(),
        error="Interrupted before it finished — the process running it was "
              "restarted. Saving a file restarts the development server, which "
              "takes any run in progress with it.",
    )
