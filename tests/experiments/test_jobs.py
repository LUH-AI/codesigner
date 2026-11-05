"""A `Job` is one execution of a run: the run opens it where its work starts
and closes it with the trials it produced. Jobs are this instance's record, so
an `.ihpo` never carries them."""

import importlib
import json

import pytest
from django.apps import apps
from django.utils import timezone

from experiments.models import Job, Run
from experiments.services import run as run_service

from tests.conftest import export_ihpo
from tests.experiments.test_cluster_backend import experiment  # noqa: F401

pytestmark = pytest.mark.django_db


def test_a_failed_run_closes_its_job_with_the_reason(experiment):  # noqa: F811
    """What: a run that fails keeps its job, ended with the error and the
    trials it had stored. How: a started local run failed by hand."""
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    run_service._start(run)

    run_service.fail_runs(Run.objects.filter(pk=run.pk), "out of memory")

    (job,) = run.jobs.all()
    assert (job.backend, job.status, job.error) == ("local", "error", "out of memory")
    assert job.finished_at is not None and job.trial_count == 0


def test_a_pending_run_has_no_job(experiment):  # noqa: F811
    """What: a job exists only once work starts. How: a created run."""
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")

    assert run.job is None


def test_the_migration_gives_every_started_run_one_job(experiment):  # noqa: F811
    """What: migration 0009 copies each started run's backend, scheduler
    number, times, status and trial range into one job, and leaves a pending
    run without one. How: its function over two runs on the current models,
    whose Run fields the migration reads are given by a stand-in model."""
    module = importlib.import_module("experiments.migrations.0009_jobs_of_runs")
    now = timezone.now()
    done = Run.objects.create(experiment=experiment, primary_metric="accuracy", status="done",
                              stopping={}, created_at=now, started_at=now, finished_at=now,
                              trial_offset=0, trial_count=4, trial_seconds=2.0)
    Run.objects.create(experiment=experiment, primary_metric="accuracy", stopping={})
    Run.backend, Run.job_id = "slurm", "4242"
    try:
        module.jobs_from_runs(apps, None)
    finally:
        del Run.backend, Run.job_id

    (job,) = Job.objects.all()
    assert job.run_id == done.pk
    assert (job.backend, job.scheduler_id, job.status) == ("slurm", "4242", "done")
    assert (job.trial_offset, job.trial_count, job.trial_seconds) == (0, 4, 2.0)


def test_a_file_carries_no_jobs(client, experiment):  # noqa: F811
    """What: an export says nothing of where the work ran. How: a run with a
    cluster job, exported."""
    run = Run.objects.create(experiment=experiment, primary_metric="accuracy", status="done",
                             stopping={}, trial_offset=0, trial_count=0)
    run.jobs.create(backend="slurm", scheduler_id="4242", status="done")

    text = export_ihpo(client, experiment.pk).content.decode()

    assert "4242" not in text and "slurm" not in text
    assert not {"jobs", "backend", "job_id", "scheduler_id"} & set(json.loads(text)["runs"][0])


def test_discarding_from_the_site_bin_removes_the_jobs(experiment, monkeypatch):  # noqa: F811
    """What: discarding an experiment deletes its runs and their jobs, and
    first cancels a scheduler job still open, which nothing would watch
    afterwards. How: a finished local job and an open cluster job, discarded
    with the scheduler stubbed."""
    from experiments.services import site as site_service

    cancelled = []
    monkeypatch.setattr(run_service, "abandon_cluster_job", cancelled.append)
    done = Run.objects.create(experiment=experiment, primary_metric="accuracy", status="done",
                              stopping={})
    done.jobs.create(backend="local", status="done", finished_at=timezone.now())
    open_run = Run.objects.create(experiment=experiment, primary_metric="accuracy",
                                  status="running", stopping={})
    open_run.jobs.create(backend="slurm", scheduler_id="555", status="running")

    site_service.discard(experiment)

    assert cancelled == ["555"]
    assert not Job.objects.exists() and not Run.objects.exists()


def test_a_job_stopped_by_somebody_ends_cancelled(experiment, monkeypatch):  # noqa: F811
    """What: an operator's Stop fails the run with the reason but ends its job
    as cancelled, telling it apart from a job that errored. How: a running
    local run stopped through `site.stop_job`, and another failed by an
    error."""
    from experiments.services import site as site_service

    stopped = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    run_service._start(stopped)
    site_service.stop_job(stopped, "Stopped by the site's operator.")
    failed = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    run_service._start(failed)
    run_service.fail_runs(Run.objects.filter(pk=failed.pk), "out of memory")

    stopped.refresh_from_db()
    assert stopped.status == "error"
    assert (stopped.job.status, stopped.job.error) == ("cancelled",
                                                       "Stopped by the site's operator.")
    assert failed.job.status == "error"
