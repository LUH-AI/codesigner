"""What a run leaves behind: enough to say what happened, however it ended.

A run is the unit the experiment's timeline is told in — when it was asked
for, when it started and stopped, which trials it produced, what happened to
it along the way — and the `.ihpo` has to carry all of it back.
"""

import json

import pytest
from django.urls import reverse
from django.utils import timezone

from tests.conftest import DATASETS_DIR, export_ihpo

pytestmark = pytest.mark.django_db


def _trials(n):
    return {
        "stats": {"submitted": n, "finished": n, "running": 0},
        "data": [{"config_id": i + 1, "cost": 0.2, "time": 1.0,
                  "scores": {"accuracy": 0.8}, "incumbent_config_id": 1}
                 for i in range(n)],
        "configs": {str(i + 1): {"max_depth": 5 + i} for i in range(n)},
        "config_origins": {}, "optimizer_state": {}, "primary_metric": "accuracy",
        "best_score": 0.8, "best_config_id": "1",
    }


def _experiment(trials=0):
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "history", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": "accuracy" if trials else None,
        "original_metric": "accuracy" if trials else None,
        "metric_names": ["accuracy"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"),
        "result": _trials(trials) if trials else None,
    }, adopt_paths=True)
    return exp


def _store(exp, n):
    from ui.models import ExperimentData

    ExperimentData.objects.filter(pk=exp.data_id).update(result=_trials(n))


def test_a_run_records_when_it_was_asked_for():
    """What: a run knows when Run was pressed, before any worker starts it.
    How: creates a run and reads its queue time."""
    from ui.services.run import create_run

    before = timezone.now()
    run = create_run(_experiment(), {"max_trials": 3}, "accuracy")

    assert run.created_at is not None and run.created_at >= before
    assert run.started_at is None


def test_a_failed_run_still_owns_the_trials_it_saved():
    """What: a run that errors after its partial writes saved trials keeps
    their range, so they are not trials of no run.
    How: starts a run on an experiment with three trials, stores two more as a
    partial write would, then fails it."""
    from ui.models import Run
    from ui.services.run import _start, create_run, fail_runs

    exp = _experiment(trials=3)
    run = create_run(exp, {"max_trials": 9}, "accuracy")
    _start(run)
    _store(exp, 5)

    fail_runs(Run.objects.filter(pk=run.pk), "it broke")
    run.refresh_from_db()

    assert (run.status, run.trial_offset, run.trial_count) == ("error", 3, 2)
    assert run.finished_at is not None and run.error == "it broke"


def test_a_run_swept_after_a_restart_is_finished_with_its_trials():
    """What: a run left "running" by a restart is closed with a finish time
    and its range, not only an error.
    How: starts a run, stores a trial, and sweeps."""
    from ui.services.run import _start, create_run, sweep_stale_runs

    exp = _experiment()
    run = create_run(exp, {"max_trials": 9}, "accuracy")
    _start(run)
    _store(exp, 1)

    assert sweep_stale_runs() == 1
    run.refresh_from_db()
    assert run.finished_at is not None
    assert (run.trial_offset, run.trial_count) == (0, 1)


def test_cancelling_records_when_and_only_the_first_time(client):
    """What: Cancel records when it was pressed; pressing it again does not
    move that. How: cancels a pending run twice through the view."""
    from ui.services.run import create_run

    exp = _experiment()
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    url = reverse("ui:run_cancel", args=[exp.pk])

    client.post(url)
    run.refresh_from_db()
    first = run.cancel_requested_at
    client.post(url)
    run.refresh_from_db()

    assert run.cancel_requested and first is not None
    assert run.cancel_requested_at == first


def test_a_metric_change_says_when_it_happened():
    """What: a run's events carry the time as well as the trial count.
    How: changes the optimized metric on an experiment with trials."""
    from ui.services.run import create_run

    exp = _experiment(trials=2)
    exp.data.metric_names = ["accuracy", "f1"]
    exp.data.save()
    run = create_run(exp, {"max_trials": 3}, "f1")

    (event,) = run.events
    assert event["kind"] == "metric_changed" and event["at_trial"] == 2
    assert event["at"]


def test_a_prior_event_keeps_when_it_took_effect_and_why_it_was_skipped():
    """What: a prior applied mid-run records the trial it took effect at, not
    the run's start; a refused one says it was refused.
    How: records two reports the optimizer could give."""
    from ui.models import Run
    from ui.services.run import _record_prior_event, create_run

    run = create_run(_experiment(), {"max_trials": 3}, "accuracy")
    _record_prior_event(run.pk, {"applied": True, "key": "codesigner",
                                 "hyperparameters": ["max_depth"], "decay": "none",
                                 "at_trial": 7, "at": "2026-10-06T10:00:00+00:00"}, 0)
    _record_prior_event(run.pk, {"applied": False, "rejected": True, "reason": "worse",
                                 "hyperparameters": ["max_depth"]}, 0)

    applied, skipped = Run.objects.get(pk=run.pk).events
    assert (applied["at_trial"], applied["at"]) == (7, "2026-10-06T10:00:00+00:00")
    assert skipped["kind"] == "prior_skipped" and skipped["rejected"] is True
    assert skipped["at"]


def test_the_cluster_records_the_prior_too(monkeypatch, settings):
    """What: a run finished on the cluster records what its prior did, as a
    local run does. How: finishes a cluster run whose status reports a prior."""
    from ui.models import Run
    from ui.services.run import _finish_cluster_run, _start, create_run

    exp = _experiment()
    run = create_run(exp, {"max_trials": 3}, "accuracy")
    _start(run)
    _finish_cluster_run(run.pk, exp.pk, {
        "state": "done", "offset": 0, "stopped_by": "max_trials", "result": _trials(3),
        "prior": {"applied": True, "key": "codesigner", "hyperparameters": ["max_depth"],
                  "decay": "none", "at_trial": 2, "at": "2026-10-06T10:00:00+00:00"}})

    (event,) = Run.objects.get(pk=run.pk).events
    assert event["kind"] == "prior_applied" and event["at_trial"] == 2


def test_runs_come_back_from_a_file_as_they_were(client):
    """What: a run's queue, start, finish and cancel times, its priors and
    trial timeout, and a run that produced no trials all survive a round trip.
    How: records two runs, exports, imports the file and compares."""
    from ui.models import Experiment, Run

    exp = _experiment(trials=3)
    now = timezone.now()
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 3}, stopped_by="max_trials",
                       created_at=now, started_at=now, finished_at=now,
                       trial_offset=0, trial_count=3,
                       priors={"max_depth": {"kind": "normal", "params": {"mu": 5}}},
                       trial_timeout={"mode": "fixed", "seconds": 30.0})
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="cancelled",
                       stopping={"max_trials": 3}, stopped_by="cancelled",
                       created_at=now, started_at=now, finished_at=now,
                       cancel_requested=True, cancel_requested_at=now,
                       trial_offset=3, trial_count=0)

    body = json.loads(export_ihpo(client, exp.pk).content)
    body["name"] = "history again"
    from django.core.files.uploadedfile import SimpleUploadedFile

    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("h.ihpo", json.dumps(body).encode())})
    again = Experiment.objects.get(data__name="history again")
    first, second = again.runs.order_by("id")

    assert first.priors == {"max_depth": {"kind": "normal", "params": {"mu": 5}}}
    assert first.trial_timeout == {"mode": "fixed", "seconds": 30.0}
    assert first.created_at == now and first.finished_at == now
    assert (second.trial_offset, second.trial_count) == (3, 0)
    assert second.cancel_requested and second.cancel_requested_at == now


def test_an_experiment_keeps_its_own_beginning_across_a_file():
    """What: the experiment's creation time travels in the file; the instance
    that imports it records when it first saw it separately.
    How: snapshots an experiment and builds another from the snapshot."""
    from ui.services import snapshot as adapter

    exp = _experiment()
    snapshot = adapter.snapshot_from_experiment(exp)
    snapshot["name"] = "copy"
    copy = adapter.experiment_from_snapshot(snapshot)

    assert snapshot["began_at"] == exp.data.created_at.isoformat()
    assert copy.data.created_at == exp.data.created_at
