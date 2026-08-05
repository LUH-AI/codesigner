"""How the background queue is configured to share itself between two jobs.

Optimizations and environment builds go through one consumer, and they have very
different shapes: a build can spend minutes downloading before anything else can
start. Both decisions that keep that from starving the instance — more than one
worker, and a higher priority for a build — are settings rather than code, so
they are pinned here with the reason attached.
"""

from django.conf import settings

from ui import tasks


def test_the_consumer_runs_more_than_one_job_at_a_time():
    """huey's default is a single worker. With one, an environment build that
    downloads a large dependency holds the queue and nothing on the instance can
    run until it finishes."""
    assert settings.HUEY["consumer"]["workers"] > 1


def test_the_workers_are_threads():
    """Both jobs wait on a subprocess almost the whole time, so a thread is the
    cheap kind of worker for them. Processes would also each need their own
    database connections and their own copy of the imports."""
    assert settings.HUEY["consumer"]["worker_type"] == "thread"


def test_preparing_an_environment_outranks_running_an_experiment():
    """A freshly uploaded model should not wait behind a queue of optimizations
    before its page stops saying "preparing"."""
    prepare = tasks.prepare_model_env_task.task_class.default_priority or 0
    run = tasks.run_experiment_task.task_class.default_priority or 0

    assert prepare > run
