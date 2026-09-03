"""Background tasks run by the huey consumer (`manage.py run_huey`).

`run_huey` autoloads each app's `tasks.py`, so defining the task here is enough
for the consumer to find it; the web process imports it lazily when enqueuing.
"""

from huey.contrib.djhuey import db_task

from .services.run import execute_run


@db_task()
def run_experiment_task(run_id):
    """Execute one run out of band. The work — rebuild from the DB, run the
    optimizer with a DB-backed cancel flag, write result and status back — lives
    in `execute_run`; this just runs it in the consumer process."""
    execute_run(run_id)


@db_task(priority=10)
def prepare_model_env_task(experiment_id):
    """Build one experiment's model environment out of band.

    Higher priority than a run: a freshly uploaded model should not wait behind
    a queue of optimizations before its page stops saying "preparing".
    """
    from .services.modelenv import prepare_environment

    prepare_environment(experiment_id)
