"""`create_run` records a run, `execute_run` rebuilds and executes it out of
band, and the page polls the status written back.
"""

import logging
import math
import random
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext as _

from access.policy import policy
from core import io, registry
from core.metrics import metric_for
from core.optimizers.base import STOPPED_BY_CANCELLED, OptimizationResult, PolledFlag
from core.optimizers import trial
from remote import jobs

from . import snapshot as snapshot_adapter
from . import trial_models
from .run_logic import apply_metrics
from .settings import eager_analytics_wanted, resolve_settings

logger = logging.getLogger(__name__)


def resolve_seed(seed_input: int) -> int:
    """Negative picks a random seed."""
    return random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)


def scoreable(metrics, model, primary_metric):
    """`scoreable` translates the refusal."""
    try:
        return trial.scoreable(metrics, model, primary_metric)
    except trial.NoProbabilities as error:
        raise ValueError(_(
            "%(metric)s is scored on class probabilities, and this model does not "
            "give them. Optimize another metric.") % {"metric": error.metric}) from error


def create_run(experiment, stopping, optimize_metric, started_by=None,
               trial_timeout=None, configurations=None):
    """`create_run` commits the optimized metric to the experiment and returns
    the pending Run. *stopping* holds at least one criterion
    (`STOPPING_CRITERIA`), and *trial_timeout* limits each model call.

    The run copies its stopping criteria, timeout and priors, because the
    experiment's own statement keeps changing.
    """
    from experiments.models import Run

    events = _metric_change_event(experiment, optimize_metric)
    current, original = apply_metrics(experiment.data.original_metric, optimize_metric)
    experiment.data.current_metric = current
    experiment.data.original_metric = original
    experiment.data.save(update_fields=["current_metric", "original_metric"])

    return Run.objects.create(
        experiment=experiment,
        primary_metric=optimize_metric,
        status="pending",
        created_at=timezone.now(),
        started_by=started_by,
        stopping=dict(stopping),
        trial_timeout=dict(trial_timeout or {}),
        # The copy happens at creation, because the reader may change the priors
        # before the run finishes.
        priors=dict(experiment.data.priors or {}),
        configurations=list(configurations or []),
        events=events,
    )


def _event(kind, at_trial, **fields) -> dict:
    return {"kind": kind, "at_trial": at_trial, "at": timezone.now().isoformat(), **fields}


def _record_prior_event(run_id, report, offset):
    """`_record_prior_event` appends the optimizer's prior report to
    `runs[].events`, at the trial where the prior took effect. `prior_skipped`
    matters as much as `prior_applied`, since `_apply_priors` continues past
    every failure, and a skip the surrogate chose is marked `rejected`. A run
    without a stated prior gets no event.
    """
    if not report:
        return

    at_trial = report.get("at_trial", offset)
    if report.get("applied"):
        event = _event("prior_applied", at_trial,
                       key=report.get("key") or "codesigner",
                       hyperparameters=report.get("hyperparameters") or [],
                       decay=report.get("decay") or "none",
                       beta=report.get("beta"))
    else:
        event = _event("prior_skipped", at_trial,
                       reason=report.get("reason") or "unknown",
                       rejected=bool(report.get("rejected")),
                       hyperparameters=report.get("hyperparameters") or [])
    if report.get("at"):
        event["at"] = report["at"]

    _record_event(run_id, event)


def _record_event(run_id, event):
    from experiments.models import Run

    run = Run.objects.filter(pk=run_id).first()
    if run is not None:
        Run.objects.filter(pk=run_id).update(events=list(run.events or []) + [event])


def _metric_change_event(experiment, now):
    """`_metric_change_event` records a metric change because the trials are
    re-read and a fitted surrogate is discarded. The old metric comes from the
    stored result, which moves only on an actual trial, because `create_run`
    commits `current_metric` even for a run that then errors.
    """
    stored = experiment.data.result or {}
    trials = stored.get("data") or []
    was = stored.get("primary_metric")
    if not was or not trials or was == now:
        return []

    optimizer = registry.OPTIMIZERS.get(experiment.data.optimizer_name)
    return [_event("metric_changed", len(trials), **{
        "from": was,
        "to": now,
        "surrogate": ("rebuilt_and_replayed"
                      if getattr(optimizer, "fits_surrogate", False) else "none"),
    })]


#: Wall-clock throttling bounds serialization cost whether trials take 0.07
#: seconds or ten minutes.
PARTIAL_RESULT_SECONDS = 5.0


def _partial_result_writer(experiment_pk, optimizer, previous_result, primary_metric):
    """`_partial_result_writer` saves the trials so far, which every trial-based
    figure can draw, and leaves the costly analytics empty as a cancelled run
    does. The database carries the update because a huey worker shares no memory
    with the page.
    """
    from experiments.models import ExperimentData

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
            # Empty dicts for the two fields without defaults.
            hyperparameter_importance={},
            hyperparameter_importance_warning={},
        )
        # The filter on the owning experiment makes the write a no-op after
        # deletion mid-run, since `ExperimentData` cascades from `Experiment`.
        ExperimentData.objects.filter(experiment__pk=experiment_pk).update(
            result=optimizer.serialize_result(partial))
        return True

    return write


def _trials_stored(experiment_pk) -> int:
    from experiments.models import ExperimentData

    result = (ExperimentData.objects.filter(experiment__pk=experiment_pk)
              .values_list("result", flat=True).first()) or {}
    return len(result.get("data") or [])


def _start(run, **fields) -> None:
    """The offset is taken at start so a failed run still owns the trials its
    partial writes saved.
    """
    from experiments.models import Run

    Run.objects.filter(pk=run.pk).update(
        status="running", started_at=timezone.now(),
        trial_offset=_trials_stored(run.experiment_id), **fields)


def fail_runs(runs, error) -> int:
    """`fail_runs` lets each run keep its trials and returns the count."""
    from experiments.models import Run

    now = timezone.now()
    count = 0
    for run in runs:
        produced = (None if run.trial_offset is None
                    else max(0, _trials_stored(run.experiment_id) - run.trial_offset))
        count += Run.objects.filter(pk=run.pk).update(
            status="error", error=error, finished_at=now, trial_count=produced)
    return count


def request_cancel(exp, user) -> None:
    """`request_cancel` stops runs after the current trial."""
    exp.runs.filter(status__in=("pending", "running"), cancel_requested=False).update(
        cancel_requested=True, cancel_requested_at=timezone.now(), cancel_requested_by=user)


def force_stop(exp) -> None:
    """`force_stop` abandons runs that ignore a cancel request."""
    stuck = exp.runs.filter(status__in=("pending", "running"), cancel_requested=True)
    for job_id in stuck.exclude(job_id="").values_list("job_id", flat=True):
        abandon_cluster_job(job_id)
    fail_runs(stuck, _("Given up on by hand: it stopped answering, and "
                       "cancelling it had no effect."))


def abandon_cluster_job(job_id: str) -> None:
    """`abandon_cluster_job` stays quiet on failure, because an unreachable
    cluster is the likely cause and freeing the row matters more. The job's
    wall-clock limit bounds it anyway.
    """
    try:
        jobs.kill(jobs.SshTransport(settings.CLUSTER_HOST, timeout=15.0), job_id)
    except Exception:  # noqa: BLE001 — see above
        logger.warning("could not scancel job %s", job_id, exc_info=True)


def execute_run(run_id):
    """The `local` backend executes in the calling process, as the test suite
    does, and `slurm` hands the optimization to a cluster that writes the same
    rows.
    """
    if settings.RUN_BACKEND == "slurm":
        return execute_run_on_cluster(run_id)
    return _execute_run_locally(run_id)


def _execute_run_locally(run_id):
    """`_execute_run_locally` rebuilds from the stored snapshot, resumes prior
    results and honours the cancel flag. Failures mark the run errored without
    crashing the caller.
    """
    from experiments.models import Run

    run = Run.objects.get(pk=run_id)
    _start(run)
    experiment = run.experiment

    try:
        # The custom-model policy is checked here as well as at the form,
        # because the executing process would run the code. A prepared custom
        # model gets its own process, and `launch` is None for a registry model
        # or an in-process upload.
        refusal = policy().custom_model_refusal(
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
        # Analytics budget for this machine; 0 means unlimited.
        optimizer.analytics_max_coalitions = (
            settings.ANALYTICS_EAGER_MAX_COALITIONS or None)
        # Hidden game figures skip the coalition evaluations entirely.
        optimizer.analytics_wanted = eager_analytics_wanted(resolve_settings(run.experiment))
        # The callback writes to the database, which `core/` cannot see.
        optimizer.progress = _partial_result_writer(
            experiment.pk, optimizer, built["result"], run.primary_metric)
        offset = len(built["result"].trials) if built["result"] else 0
        cancel = PolledFlag(
            lambda: Run.objects.filter(pk=run_id, cancel_requested=True).exists())
        # Only an in-process model can hand a fit over.
        keeper = (trial_models.keeper_for(run, built, metric_for(run.primary_metric))
                  if launch is None else None)
        optimizer.keep_model = keeper

        def _optimize(model):
            _remember_config_space(experiment.pk, model, built["seed"])
            return optimizer.optimize(
                model,
                built["X_train"], built["y_train"], built["X_val"], built["y_val"],
                metrics=scoreable(built["metrics"], model, run.primary_metric),
                primary_metric=run.primary_metric,
                previous_result=built["result"],
                seed=built["seed"],
                cancel_event=cancel,
                stopping=run.stopping,
                splits=built["splits"],
                # Passed in, since `core/` imports no Django.
                priors=experiment.data.priors or None,
                # Page-picked configurations run before the search proposes any.
                configurations=run.configurations or None,
            )

        if launch is None:
            result = _optimize(built["model"])
        else:
            from core.modelhost import model_session

            from . import modelenv

            with model_session(
                launch, built["splits"],
                seed=built["seed"], task=built["task"],
                feature_kinds=built["feature_kinds"], forecast=built["forecast"],
                cancel=cancel,
                **modelenv.session_kwargs(run.trial_timeout, seed=built["seed"]),
            ) as model:
                result = _optimize(model)
    except Exception as exc:  # noqa: BLE001 — any failure is reported on the run
        fail_runs(Run.objects.filter(pk=run_id), str(exc))
        return

    from experiments.models import ExperimentData

    if keeper is not None and keeper.no_room:
        _record_event(run_id, _event("trial_models_not_kept", offset,
                                     count=keeper.no_room, reason="no room"))
    cancelled = Run.objects.filter(pk=run_id, cancel_requested=True).exists()
    if result.trials:
        # A filtered update keeps a deleted experiment deleted.
        ExperimentData.objects.filter(experiment__pk=experiment.pk).update(
            result=optimizer.serialize_result(result),
        )
    # The run box shows the count, time and overhead of the trials this run
    # added.
    new_trials = result.trials[offset:]
    _record_prior_event(run_id, (getattr(result, "metadata", None) or {}).get("prior"), offset)
    Run.objects.filter(pk=run_id).update(
        status="cancelled" if cancelled else "done", finished_at=timezone.now(),
        trial_seconds=sum(t.duration for t in new_trials),
        trial_count=len(new_trials),
        trial_offset=offset,
        # A cancel counts as the stop reason.
        stopped_by=(STOPPED_BY_CANCELLED if cancelled
                    else (result.metadata.get("stopped_by") or "")),
    )


def execute_run_on_cluster(run_id):
    """`execute_run_on_cluster` relays the job's files, since the job has no
    route to the application database. `partial.json` becomes the live result, a
    Cancel writes a `CANCEL` file that keeps the trials, and `result.json` and
    `status.json` become the final writes. Errors are recorded on the run, as on
    the local path.
    """
    from experiments.models import Run
    run = Run.objects.get(pk=run_id)
    experiment = run.experiment
    _start(run, backend="slurm")

    try:
        dataset = experiment.data.dataset_path()
        if dataset is None:
            raise RuntimeError(
                "this experiment has no dataset stored, so there is nothing to "
                "send to the cluster")

        ssh = jobs.transport()
        snapshot = snapshot_adapter.snapshot_from_experiment(experiment)
        workdir = jobs.stage(
            ssh, run_id, snapshot,
            {
                "stopping": run.stopping,
                "primary_metric": run.primary_metric,
                "trial_timeout": run.trial_timeout,
                "analytics_max_coalitions": settings.ANALYTICS_EAGER_MAX_COALITIONS,
                "analytics_wanted": eager_analytics_wanted(resolve_settings(experiment)),
                # Priors travel with the other run bounds.
                "priors": experiment.data.priors or None,
                "configurations": run.configurations or None,
            },
            dataset=dataset,
            model=Path(experiment.data.model_file.path) if experiment.data.model_file else None,
        )
        job_id = jobs.submit(ssh, run_id, workdir, hours=_hours_for(run))
        Run.objects.filter(pk=run_id).update(job_id=job_id)

        status = _watch(ssh, run_id, experiment.pk, workdir, job_id)
    except Exception as exc:  # noqa: BLE001 — any failure is reported on the run
        fail_runs(Run.objects.filter(pk=run_id), str(exc))
        return

    _finish_cluster_run(run_id, experiment.pk, status)


def _watch(ssh, run_id, experiment_pk, workdir, job_id) -> dict:
    """`_watch` returns the job's `status.json`, or an error status when the
    scheduler killed the job before it wrote one.
    """
    from experiments.models import ExperimentData, Run
    cancelled = False
    while True:
        state = jobs.poll(ssh, job_id)

        if not cancelled and Run.objects.filter(
                pk=run_id, cancel_requested=True).exists():
            # Requested once; the run still writes its trials.
            jobs.request_cancel(ssh, workdir)
            cancelled = True

        if state in (jobs.DONE, jobs.FAILED, jobs.GONE):
            break

        # Queued jobs have written nothing, so polling waits for RUNNING.
        if state == jobs.RUNNING:
            partial = jobs.fetch(ssh, workdir, "partial.json")
            if partial:
                ExperimentData.objects.filter(
                    experiment__pk=experiment_pk).update(result=partial)

        time.sleep(settings.CLUSTER_POLL_SECONDS)

    status = jobs.fetch(ssh, workdir, "status.json")
    if status is None:
        reason = jobs.tail_error(ssh, workdir)
        status = {"state": "error", "offset": 0, "stopped_by": "",
                  "config_space": None,
                  "error": (f"the job ended without finishing ({state})"
                            + (f": {reason}" if reason else ""))}
    status["result"] = jobs.fetch(ssh, workdir, "result.json")
    return status


def _finish_cluster_run(run_id, experiment_pk, status: dict) -> None:
    """`_finish_cluster_run` writes the same fields with the same filtered
    `update()` as the local path.
    """
    from experiments.models import ExperimentData, Run

    if status["state"] == "error":
        fail_runs(Run.objects.filter(pk=run_id),
                  status.get("error") or "the run failed on the cluster")
        return

    result = status.get("result")
    trials = (result or {}).get("data") or []
    offset = int(status.get("offset") or 0)
    new_trials = trials[offset:]

    fields = {"result": result} if result else {}
    if status.get("config_space"):
        fields["config_space"] = status["config_space"]
    if fields:
        ExperimentData.objects.filter(experiment__pk=experiment_pk).update(**fields)

    _record_prior_event(run_id, status.get("prior"), offset)
    Run.objects.filter(pk=run_id).update(
        status="cancelled" if status["state"] == "cancelled" else "done",
        finished_at=timezone.now(),
        trial_seconds=sum(float(t.get("time") or 0.0) for t in new_trials),
        trial_count=len(new_trials),
        trial_offset=offset,
        stopped_by=(STOPPED_BY_CANCELLED if status["state"] == "cancelled"
                    else (status.get("stopped_by") or "")),
    )


def _hours_for(run) -> int:
    """`_hours_for` rounds the run's time limit up and adds an hour, else uses
    the deployment ceiling. An outlived allocation kills the job mid-write and
    loses trials.
    """
    seconds = (run.stopping or {}).get("max_seconds")
    if not seconds:
        return settings.CLUSTER_MAX_HOURS
    return max(1, math.ceil(float(seconds) / 3600.0) + 1)


def _remember_config_space(experiment_pk, model, seed) -> None:
    """The run is the only moment the space is certainly available, since a
    read-only page never imports a custom model. A model that cannot describe
    its space leaves the stored one alone.
    """
    from experiments.models import ExperimentData

    try:
        space = model.get_config_space(seed=seed).to_serialized_dict()
    except Exception:  # noqa: BLE001 — a run must not fail over its own bookkeeping
        return
    ExperimentData.objects.filter(experiment__pk=experiment_pk).update(
        config_space=space)


def _model_launch(experiment):
    """`_model_launch` gives ``(None, "")`` for an in-process import, and a
    refusal string blocks the run.
    """
    from . import modelenv

    if not experiment.data.model_file:
        return None, ""
    return modelenv.resolve_runner(experiment)


def start_background_run(run_id):
    """The lazy import avoids an import cycle with tasks.py; dispatch.py
    explains the immediate-mode thread.
    """
    from .. import tasks
    from .dispatch import enqueue

    enqueue(tasks.run_experiment_task, run_id)


def sweep_stale_runs():
    """The durable huey queue keeps pending runs, so only running runs count as
    stale.
    """
    from experiments.models import Run

    return fail_runs(Run.objects.filter(status="running"), "Interrupted by a restart.")


#: Immediate-mode runs of this process all start later.
PROCESS_STARTED = timezone.now()

#: Once per process; ui/apps.py explains why `ready()` skips the database.
_SWEPT = False


def sweep_orphaned_runs():
    """`sweep_orphaned_runs` applies only in immediate mode, where runs execute
    on daemon threads that die when `runserver` restarts. Such a run, if older
    than the process, has no executor, and nobody polls its cancel flag.
    Consumer mode relies on the `sweep_stale_runs` command instead.
    """
    global _SWEPT
    from huey.contrib.djhuey import HUEY

    from experiments.models import Run

    if _SWEPT or not HUEY.immediate:
        return 0
    _SWEPT = True
    return fail_runs(
        Run.objects.filter(status="running", started_at__lt=PROCESS_STARTED),
        "Interrupted before it finished — the process running it was restarted. "
        "Saving a file restarts the development server, which takes any run in "
        "progress with it.")
