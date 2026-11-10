"""`create_run` records a run; `execute_run` executes it out of band while the
page polls.
"""

import logging
import math
import random
import time
from pathlib import Path

from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from django.utils.translation import gettext as _

from core import io, registry
from core.metrics import metric_for
from core.optimizers.base import STOPPED_BY_CANCELLED, OptimizationResult, PolledFlag
from core.optimizers import trial
from remote import jobs

from . import history
from . import snapshot as snapshot_adapter
from . import trial_models
from .run_logic import apply_metrics
from .settings import eager_analytics_wanted, resolve_settings

logger = logging.getLogger(__name__)


def resolve_seed(seed_input: int) -> int:
    """Negatives pick randomly."""
    return random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)


def scoreable(metrics, model, primary_metric):
    try:
        return trial.scoreable(metrics, model, primary_metric)
    except trial.NoProbabilities as error:
        raise ValueError(_(
            "%(metric)s is scored on class probabilities, and this model does not "
            "give them. Optimize another metric.") % {"metric": error.metric}) from error


def create_run(experiment, stopping, optimize_metric, started_by=None,
               trial_timeout=None, configurations=None):
    """`create_run` commits the optimized metric, records the priors the run
    takes up, and returns the pending Run. *stopping* needs at least one
    criterion.
    """
    from experiments.models import Run

    events = _metric_change_event(experiment, optimize_metric)
    current, original = apply_metrics(experiment.data.original_metric, optimize_metric)
    experiment.data.current_metric = current
    experiment.data.original_metric = original
    experiment.data.save(update_fields=["current_metric", "original_metric"])

    run = Run.objects.create(
        experiment=experiment,
        primary_metric=optimize_metric,
        status="pending",
        created_at=timezone.now(),
        started_by=started_by,
        stopping=dict(stopping),
        trial_timeout=dict(trial_timeout or {}),
        # The copy lets the reader edit the priors mid-run.
        priors=dict(experiment.data.priors or {}),
        optimizer_params=dict(experiment.data.optimizer_params or {}),
        configurations=list(configurations or []),
        events=events,
    )
    history.record_run_priors(run)
    return run


def _event(kind, at_trial, **fields) -> dict:
    return {"kind": kind, "at_trial": at_trial, "at": timezone.now().isoformat(), **fields}


def _record_prior_event(run_id, report, offset):
    """`_record_prior_event` appends the prior report to `runs[].events`.
    `.apply_priors` continues past every failure, so `prior_skipped` events
    matter; `rejected` marks a skip the surrogate chose.
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
    """A metric change discards a fitted surrogate. The old metric comes from
    the stored result, since `create_run` commits `current_metric` even for a
    failed run.
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


#: Wall-clock throttling bounds serialization cost for any trial length.
PARTIAL_RESULT_SECONDS = 5.0


def _partial_result_writer(experiment_pk, optimizer, previous_result, primary_metric):
    """`_partial_result_writer` saves the trials so far with empty analytics, as
    a cancelled run does. Huey workers share no memory with the page.
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
            hyperparameter_importance={},
            hyperparameter_importance_warning={},
        )
        # The filter skips the write after a mid-run deletion.
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
    """The start offset lets a failed run own its partial trials. A local run
    is one job, in the worker.
    """
    from experiments.models import Run

    now, offset = timezone.now(), _trials_stored(run.experiment_id)
    Run.objects.filter(pk=run.pk).update(
        status="running", started_at=now, trial_offset=offset, **fields)
    _open_job(run.pk, "local", status="running", submitted_at=now, started_at=now,
              trial_offset=offset)


def _open_job(run_id, backend, **fields):
    from experiments.models import Job

    return Job.objects.create(run_id=run_id, backend=backend, **fields)


def _close_jobs(run, status, error="") -> None:
    """`_close_jobs` ends *run*'s unfinished jobs with *status*, each with the
    trials stored since its own offset and their seconds."""
    from experiments.models import ExperimentData, Job

    result = (ExperimentData.objects.filter(experiment__pk=run.experiment_id)
              .values_list("result", flat=True).first()) or {}
    trials = result.get("data") or []
    for job in Job.objects.filter(run_id=run.pk, finished_at__isnull=True):
        produced = None if job.trial_offset is None else trials[job.trial_offset:]
        Job.objects.filter(pk=job.pk).update(
            status=status, error=error, finished_at=timezone.now(),
            trial_count=None if produced is None else len(produced),
            trial_seconds=(None if produced is None
                           else sum(float(t.get("time") or 0.0) for t in produced)))


def fail_runs(runs, error, *, jobs_end="error") -> int:
    """`fail_runs` keeps trials and returns a count. The runs' open jobs end
    with *jobs_end*: "error", or "cancelled" for somebody who stopped them."""
    from experiments.models import Run

    now = timezone.now()
    count = 0
    for run in runs:
        produced = (None if run.trial_offset is None
                    else max(0, _trials_stored(run.experiment_id) - run.trial_offset))
        count += Run.objects.filter(pk=run.pk).update(
            status="error", error=error, finished_at=now, trial_count=produced)
        _close_jobs(run, jobs_end, error)
    return count


def request_cancel(exp, user) -> None:
    """`request_cancel` stops runs after the current trial."""
    from experiments.models import Run

    exp.runs.filter(status__in=Run.ACTIVE_STATUSES, cancel_requested=False).update(
        cancel_requested=True, cancel_requested_at=timezone.now(), cancel_requested_by=user)


def force_stop(exp) -> None:
    """`force_stop` abandons runs that ignore a cancel request."""
    from experiments.models import Run

    stuck = exp.runs.filter(status__in=Run.ACTIVE_STATUSES, cancel_requested=True)
    abandon_jobs(stuck)
    fail_runs(stuck, _("Given up on by hand: it stopped answering, and "
                       "cancelling it had no effect."), jobs_end="cancelled")


def abandon_jobs(runs) -> None:
    """`abandon_jobs` cancels the scheduler jobs still open for *runs*."""
    from experiments.models import Job

    for scheduler_id in (Job.objects.filter(run__in=runs, finished_at__isnull=True)
                         .exclude(scheduler_id="").values_list("scheduler_id", flat=True)):
        abandon_cluster_job(scheduler_id)


def abandon_cluster_job(job_id: str) -> None:
    """`abandon_cluster_job` logs failures, since an unreachable cluster is the
    likely cause. The job's wall-clock limit bounds it anyway.
    """
    try:
        jobs.kill(jobs.SshTransport(settings.CLUSTER_HOST, timeout=15.0), job_id)
    except Exception:  # noqa: BLE001 — see above
        logger.warning("could not scancel job %s", job_id, exc_info=True)


def execute_run(run_id):
    """`local` executes the run in-process; `slurm` sends it to a cluster that
    writes the same rows.
    """
    if settings.RUN_BACKEND == "slurm":
        return execute_run_on_cluster(run_id)
    return _execute_run_locally(run_id)


def _execute_run_locally(run_id):
    """`_execute_run_locally` rebuilds from the stored snapshot and marks a
    failed run errored without raising.
    """
    from experiments.models import Run

    run = Run.objects.get(pk=run_id)
    _start(run)
    experiment = run.experiment

    try:
        # The executing process checks the custom-model policy again, since it
        # would run the code. `launch` is None for a registry model or an
        # in-process upload.
        refusal = _model_refusal(run)
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
        # 0 means unlimited.
        optimizer.analytics_max_coalitions = (
            settings.ANALYTICS_EAGER_MAX_COALITIONS or None)
        # Hidden games skip coalition evaluations.
        optimizer.analytics_wanted = eager_analytics_wanted(resolve_settings(run.experiment))
        # `core/` cannot see the database.
        optimizer.progress = _partial_result_writer(
            experiment.pk, optimizer, built["result"], run.primary_metric)
        offset = len(built["result"].trials) if built["result"] else 0
        cancel = PolledFlag(
            lambda: Run.objects.filter(pk=run_id, cancel_requested=True).exists())
        # Out-of-process models keep no fit.
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
                # `core/` imports no Django.
                priors=experiment.data.priors or None,
                # Page-picked configurations run first.
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
        ExperimentData.objects.filter(experiment__pk=experiment.pk).update(
            result=optimizer.serialize_result(result),
        )
    # The run box only counts this run's trials.
    new_trials = result.trials[offset:]
    _record_prior_event(run_id, (getattr(result, "metadata", None) or {}).get("prior"), offset)
    Run.objects.filter(pk=run_id).update(
        status="cancelled" if cancelled else "done", finished_at=timezone.now(),
        trial_seconds=sum(t.duration for t in new_trials),
        trial_count=len(new_trials),
        trial_offset=offset,
        stopped_by=(STOPPED_BY_CANCELLED if cancelled
                    else (result.metadata.get("stopped_by") or "")),
    )
    _close_jobs(run, "cancelled" if cancelled else "done")


def execute_run_on_cluster(run_id):
    """`execute_run_on_cluster` relays the job's files, since the job cannot
    reach the database. A Cancel writes a `CANCEL` file that keeps the trials.
    """
    from experiments.models import Run
    run = Run.objects.get(pk=run_id)
    experiment = run.experiment
    Run.objects.filter(pk=run_id).update(trial_offset=_trials_stored(experiment.pk))

    try:
        # The job runs the code on the cluster, so the policy check comes first.
        refusal = _model_refusal(run)
        if refusal:
            raise RuntimeError(refusal)
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
                "priors": experiment.data.priors or None,
                "configurations": run.configurations or None,
            },
            dataset=dataset,
            model=Path(experiment.data.model_file.path) if experiment.data.model_file else None,
        )
        job_id = jobs.submit(ssh, run_id, workdir, hours=_hours_for(run))
        Run.objects.filter(pk=run_id).update(status="submitted")
        _open_job(run_id, "slurm", scheduler_id=job_id, status="submitted",
                  submitted_at=timezone.now(), trial_offset=_trials_stored(experiment.pk))

        status = _watch(ssh, run_id, experiment.pk, workdir, job_id)
    except Exception as exc:  # noqa: BLE001 — any failure is reported on the run
        fail_runs(Run.objects.filter(pk=run_id), str(exc))
        return

    _finish_cluster_run(run_id, experiment.pk, status)
    # With the results stored here, the copies on the cluster are deleted.
    try:
        jobs.remove(ssh, workdir)
    except Exception:  # noqa: BLE001 — an unreachable cluster keeps them until a retry
        logger.warning("could not remove %s from the cluster", workdir, exc_info=True)


def _watch(ssh, run_id, experiment_pk, workdir, job_id) -> dict:
    """`_watch` returns `status.json`, or an error status for a killed job.
    """
    from experiments.models import ExperimentData, Job, Run
    cancelled = False
    started = False
    while True:
        state = jobs.poll(ssh, job_id)

        if not started and state != jobs.QUEUED:
            now = timezone.now()
            Run.objects.filter(pk=run_id, status="submitted").update(
                status="running", started_at=now)
            Job.objects.filter(run_id=run_id, scheduler_id=job_id, status="submitted").update(
                status="running", started_at=now)
            started = True

        if not cancelled and Run.objects.filter(
                pk=run_id, cancel_requested=True).exists():
            # Cancelled jobs still write trials.
            jobs.request_cancel(ssh, workdir)
            cancelled = True

        if state in (jobs.DONE, jobs.FAILED, jobs.GONE):
            break

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
    run = Run.objects.get(pk=run_id)
    Run.objects.filter(pk=run_id).update(
        status="cancelled" if status["state"] == "cancelled" else "done",
        finished_at=timezone.now(),
        trial_seconds=sum(float(t.get("time") or 0.0) for t in new_trials),
        trial_count=len(new_trials),
        trial_offset=offset,
        stopped_by=(STOPPED_BY_CANCELLED if status["state"] == "cancelled"
                    else (status.get("stopped_by") or "")),
    )
    _close_jobs(run, "cancelled" if status["state"] == "cancelled" else "done")


def _hours_for(run) -> int:
    """`_hours_for` adds an hour to the rounded-up time limit, since an outlived
    allocation kills the job mid-write.
    """
    seconds = (run.stopping or {}).get("max_seconds")
    if not seconds:
        return settings.CLUSTER_MAX_HOURS
    return max(1, math.ceil(float(seconds) / 3600.0) + 1)


def _remember_config_space(experiment_pk, model, seed) -> None:
    """A read-only page never imports a custom model, so the run stores the
    space. A failing model leaves the stored space alone.
    """
    from experiments.models import ExperimentData

    try:
        space = model.get_config_space(seed=seed).to_serialized_dict()
    except Exception:  # noqa: BLE001 — a run must not fail over its own bookkeeping
        return
    ExperimentData.objects.filter(experiment__pk=experiment_pk).update(
        config_space=space)


def _model_refusal(run) -> str:
    from . import modelenv

    return modelenv.refusal(run.experiment, run.started_by or run.experiment.owner)


def _model_launch(experiment):
    """`_model_launch` gives `(None, "")` for an in-process import; a refusal
    string blocks the run."""
    from . import modelenv

    if not experiment.data.model_file:
        return None, ""
    return modelenv.resolve_runner(experiment)


def start_background_run(run_id):
    """The lazy import avoids a cycle with `.tasks`.
    """
    from .. import tasks
    from .dispatch import enqueue

    enqueue(tasks.run_experiment_task, run_id)


def sweep_stale_runs():
    """The durable huey queue keeps pending runs.
    """
    from experiments.models import Run

    return fail_runs(Run.objects.filter(status__in=("submitted", "running")),
                     "Interrupted by a restart.")


PROCESS_STARTED = timezone.now()

#: Sweeps once per process; `ui/apps.py` explains why.
_SWEPT = False


def sweep_orphaned_runs():
    """`sweep_orphaned_runs` only applies in immediate mode, where runs execute
    on daemon threads that die when `runserver` restarts. Consumer mode uses the
    `sweep_stale_runs` command instead.
    """
    global _SWEPT
    from huey.contrib.djhuey import HUEY

    from experiments.models import Run

    if _SWEPT or not HUEY.immediate:
        return 0
    _SWEPT = True
    return fail_runs(
        Run.objects.filter(Q(status="running", started_at__lt=PROCESS_STARTED)
                           | Q(status="submitted", created_at__lt=PROCESS_STARTED)),
        "Interrupted before it finished — the process running it was restarted. "
        "Saving a file restarts the development server, which takes any run in "
        "progress with it.")
