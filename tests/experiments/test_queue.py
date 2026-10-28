"""Optimizations and environment builds share one huey consumer, and a build can
download for minutes.
"""

from django.conf import settings

from experiments import tasks


def test_the_consumer_runs_more_than_one_job_at_a_time():
    """With huey's single default worker, one large build download blocks the
    queue."""
    assert settings.HUEY["consumer"]["workers"] > 1


def test_the_workers_are_threads():
    """Threads suit both jobs, since each one waits on a subprocess."""
    assert settings.HUEY["consumer"]["worker_type"] == "thread"


def test_preparing_an_environment_outranks_running_an_experiment():
    """A new model's page should leave "preparing" without waiting behind
    optimizations."""
    prepare = tasks.prepare_model_env_task.task_class.default_priority or 0
    run = tasks.run_experiment_task.task_class.default_priority or 0

    assert prepare > run
