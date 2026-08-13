"""After a run, the run box notes total wall-clock duration and overhead.

Overhead = total wall time − Σ this run's trial durations (the search/bookkeeping
time SMAC and Codesigner spent between evaluations). The per-run trial-second
sum is stored on the Run; the total comes from its timestamps.
"""

import datetime

import pytest
from django.urls import reverse
from django.utils import timezone

from tests.conftest import DATASETS_DIR


def test_run_duration_property():
    from ui.models import Experiment, Run
    exp = Experiment.objects.create(name="d", model_name="Random Forest",
                                    optimizer_name="Random Search", metric_names=["accuracy"], seed=0)
    start = timezone.now()
    run = Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
                             status="done", started_at=start,
                             finished_at=start + datetime.timedelta(seconds=5))
    assert run.duration == pytest.approx(5.0, abs=0.01)


def test_execute_run_stores_trial_seconds():
    from ui.services import snapshot as adapter
    from ui.services.run import create_run, execute_run
    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "rs", "model_name": "Random Forest", "model_path": "",
        "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    execute_run(run.id)
    run.refresh_from_db()
    assert run.status == "done"
    assert run.trial_seconds is not None and run.trial_seconds >= 0.0
    assert run.trial_count == 3


def test_detail_page_shows_run_summary(client):
    """A finished run surfaces total duration and overhead on the detail page."""
    from ui.models import Experiment, Run
    result = {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.2, "time": 2.0,
                  "scores": {"accuracy": 0.8, "f1": 0.7, "precision": 0.75, "recall(macro)": 0.72},
                  "incumbent_score": 0.8, "incumbent_config_id": 1}],
        "configs": {"1": {"n_estimators": 100}}, "config_origins": {"1": "Random Search"},
        "optimizer_state": {}, "primary_metric": "accuracy", "best_score": 0.8,
        "best_config_id": "1", "hyperparameter_importance": {},
        "hyperparameter_importance_warning": {}, "trials_limit": None,
    }
    exp = Experiment.objects.create(
        name="sum", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1", "precision", "recall(macro)"],
        current_metric="accuracy", original_metric="accuracy", seed=0, result=result)
    start = timezone.now()
    Run.objects.create(experiment=exp, stopping={"max_trials": 1}, primary_metric="accuracy", status="done",
                       started_at=start, finished_at=start + datetime.timedelta(seconds=3),
                       trial_seconds=2.0, trial_count=1)
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "overhead" in body.lower()
    assert "3.0" in body and "2.0" in body   # total 3s, 2s in trials
    assert "1 trials" in body                # count of trials performed
