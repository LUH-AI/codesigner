"""`run_huey` autoloads each app's `tasks.py`, so a task defined here needs no
registration.
"""

from huey.contrib.djhuey import db_task

from .services.run import execute_run


@db_task()
def run_experiment_task(run_id):
    execute_run(run_id)


@db_task(priority=10)
def prepare_model_env_task(experiment_id):
    """Priority 10 lets a new model skip queued runs.
    """
    from .services.modelenv import prepare_environment

    prepare_environment(experiment_id)
