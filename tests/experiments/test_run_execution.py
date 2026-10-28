"""`test_run_execution` pins the database wiring of `create_run` and
`execute_run` on small, deterministic iris runs.
"""

import pytest

from core import io

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _make_experiment(metric_names=None, primary=None, original=None):
    """`_make_experiment` saves an iris experiment without a result."""
    from experiments.services import snapshot as adapter

    snapshot = {
        "version": "0.1.0",
        "name": "run-exp",
        "model_name": "Random Forest",
        "model_path": "",
        "optimizer_name": "Random Search",
        "optimizer_params": {},
        "primary_metric": primary,
        "original_metric": original,
        "metric_names": metric_names or ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"),
        "result": None,
    }
    return adapter.experiment_from_snapshot(snapshot, adopt_paths=True)


@pytest.mark.django_db
def test_create_run_records_pending_run_and_commits_first_metric():
    """`create_run` makes a pending run and pins the chosen metric as primary
    and original.
    """
    from experiments.services.run import create_run

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 3}, "f1")

    assert run.status == "pending"
    assert run.cancel_requested is False
    assert run.max_trials == 3
    assert run.primary_metric == "f1"

    exp.refresh_from_db()
    assert exp.data.current_metric == "f1"
    assert exp.data.original_metric == "f1"


@pytest.mark.django_db
def test_create_run_metric_change_moves_primary_not_original():
    from experiments.services.run import create_run

    exp = _make_experiment(primary="accuracy", original="accuracy")
    create_run(exp, {"max_trials": 2}, "f1")

    exp.refresh_from_db()
    assert exp.data.current_metric == "f1"
    assert exp.data.original_metric == "accuracy"


@pytest.mark.django_db
def test_execute_run_completes_and_stores_result():
    """The run ends "done" with both timestamps and a 3-trial result.
    """
    from experiments.services.run import create_run, execute_run

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    exp.refresh_from_db()
    assert run.status == "done"
    assert run.started_at is not None and run.finished_at is not None
    assert exp.data.result is not None
    assert len(exp.data.result["data"]) == 3


@pytest.mark.django_db
def test_execute_run_resumes_from_previous_result():
    """After runs of 3 and 2 trials, `trial_offset` numbers the stored trials 1
    to 5.
    """
    from experiments.services.run import create_run, execute_run

    exp = _make_experiment()
    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    execute_run(create_run(exp, {"max_trials": 2}, "accuracy").id)

    exp.refresh_from_db()
    assert exp.data.result is not None
    trials = exp.data.result["data"]
    assert [t["config_id"] for t in trials] == [1, 2, 3, 4, 5]


@pytest.mark.django_db
def test_execute_run_honours_a_preset_cancel():
    """A preset `cancel_requested` stops the run at the first check, independent
    of optimizer speed.
    """
    from experiments.models import Run
    from experiments.services.run import create_run, execute_run

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 5}, "accuracy")
    Run.objects.filter(pk=run.id).update(cancel_requested=True)

    execute_run(run.id)

    run.refresh_from_db()
    assert run.status == "cancelled"
    assert run.finished_at is not None


@pytest.mark.django_db
def test_execute_run_records_errors():
    """An unknown metric fails the rebuild, and the run records the error.
    """
    from experiments.services.run import create_run, execute_run

    exp = _make_experiment(metric_names=["accuracy", "bogus-metric"])
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    assert run.status == "error"
    assert run.error


@pytest.mark.django_db
def test_a_polled_flag_reflects_the_database():
    """`PolledFlag` reports the row's current `cancel_requested`."""
    from core.optimizers.base import PolledFlag
    from experiments.models import Run

    exp = _make_experiment()
    from experiments.services.run import create_run
    run = create_run(exp, {"max_trials": 3}, "accuracy")

    flag = PolledFlag(
        lambda: Run.objects.filter(pk=run.id, cancel_requested=True).exists(), ttl=0)
    assert flag.is_set() is False
    Run.objects.filter(pk=run.id).update(cancel_requested=True)
    assert flag.is_set() is True
