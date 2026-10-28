"""`RUN_BACKEND=slurm` stages a run on a cluster, follows its Slurm job and
writes back the rows a local run writes. A transport double with canned
scheduler output replaces the cluster.
"""

import json

import pytest
from django.urls import reverse

from remote import jobs

from tests.conftest import DATASETS_DIR

pytestmark = pytest.mark.django_db


class FakeTransport:
    """`FakeTransport` answers from a script instead of a scheduler. Successive
    polls pop `states`, and `get` reads back whatever `files` holds.
    """

    def __init__(self, states=("done",), files=None, submit="81234"):
        self.host = "fake"
        self.files: dict[str, bytes] = dict(files or {})
        self.commands: list[str] = []
        self._states = list(states)
        self._submit = submit

    def put(self, data: bytes, remote: str) -> None:
        self.files[remote] = data

    def put_file(self, local, remote: str) -> None:
        self.files[remote] = local.read_bytes()

    def get(self, remote: str):
        return self.files.get(remote)

    def run(self, command: str):
        self.commands.append(command)
        if command.startswith("sbatch") or " sbatch " in command:
            return jobs.Command(0, self._submit + "\n", "")
        if command.startswith("squeue"):
            state = self._states[0] if len(self._states) == 1 else self._states.pop(0)
            return jobs.Command(0, {"queued": "PENDING\n", "running": "RUNNING\n"}.get(state, ""), "")
        if command.startswith("sacct"):
            state = self._states[-1]
            return jobs.Command(
                0, {"done": "COMPLETED\n", "failed": "FAILED\n", "gone": ""}.get(state, "COMPLETED\n"), "")
        if command.startswith("tail"):
            return jobs.Command(0, "slurmstepd: error: exceeded memory limit\n", "")
        return jobs.Command(0, "", "")


@pytest.fixture
def cluster_settings(settings):
    settings.RUN_BACKEND = "slurm"
    settings.CLUSTER_HOST = "fake"
    settings.CLUSTER_ROOT = "codesigner"
    settings.CLUSTER_PARTITION = "kisski-inference"
    settings.CLUSTER_POLL_SECONDS = 0
    return settings


@pytest.fixture
def experiment():
    from experiments.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "remote", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 3,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def _finished_result(trials=3):
    """`_finished_result` mimics a job's result.json."""
    return {
        "stats": {"submitted": trials, "finished": trials, "running": 0},
        "data": [{"config_id": i + 1, "cost": 0.2, "time": 1.5,
                  "scores": {"accuracy": 0.8}, "incumbent_config_id": 1}
                 for i in range(trials)],
        "configs": {str(i + 1): {"max_depth": 5 + i} for i in range(trials)},
        "config_origins": {}, "optimizer_state": {}, "primary_metric": "accuracy",
        "best_score": 0.8, "best_config_id": "1",
    }


def _run(experiment, transport, monkeypatch, states=("done",), status=None,
         result=None):
    from experiments.services import run as run_service

    workdir = None

    def _fake_transport():
        return transport

    monkeypatch.setattr(jobs, "transport", _fake_transport)

    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = jobs.workdir_for(run.id)
    if status is not None:
        transport.files[f"{workdir}/status.json"] = json.dumps(status).encode()
    if result is not None:
        transport.files[f"{workdir}/result.json"] = json.dumps(result).encode()
    transport._states = list(states)
    run_service.execute_run(run.id)
    run.refresh_from_db()
    return run, workdir


def test_everything_the_job_needs_is_staged(cluster_settings, experiment, monkeypatch):
    """The snapshot's dataset path exists only here, so the job needs the staged
    file.
    """
    transport = FakeTransport()
    run, workdir = _run(experiment, transport, monkeypatch,
                        status={"state": "done", "offset": 0, "stopped_by": "max_trials",
                                "config_space": None, "error": ""},
                        result=_finished_result())

    assert f"{workdir}/snapshot.json" in transport.files
    assert f"{workdir}/run.json" in transport.files
    staged = transport.files[f"{workdir}/dataset.csv"]
    assert staged == experiment.data.dataset_path().read_bytes(), "the job got a different dataset"

    sent = json.loads(transport.files[f"{workdir}/run.json"])
    assert sent["stopping"] == {"max_trials": 3}
    assert sent["primary_metric"] == "accuracy"


def test_the_job_id_is_recorded_before_the_run_is_watched(cluster_settings,
                                                          experiment, monkeypatch):
    """A restarted consumer can then find the job again."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 0, "stopped_by": "",
                          "config_space": None, "error": ""},
                  result=_finished_result())

    assert run.job_id == "81234"
    assert run.backend == "slurm"


def test_a_finished_job_writes_the_rows_a_local_run_writes(cluster_settings,
                                                           experiment, monkeypatch):
    """The history cannot tell where a run happened."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 0, "stopped_by": "max_trials",
                          "config_space": None, "error": ""},
                  result=_finished_result(trials=3))
    experiment.refresh_from_db()

    assert run.status == "done"
    assert run.stopped_by == "max_trials"
    assert run.trial_count == 3
    assert run.trial_seconds == pytest.approx(4.5)
    assert len(experiment.data.result["data"]) == 3


def test_the_search_space_the_job_saw_is_kept(cluster_settings, experiment, monkeypatch):
    """Surrogate-backed figures need the space only the job could read."""
    space = {"name": None, "hyperparameters": [], "conditions": [],
             "forbiddens": [], "python_module_version": "1.2.0",
             "format_version": 0.4}
    _run(experiment, FakeTransport(), monkeypatch,
         status={"state": "done", "offset": 0, "stopped_by": "",
                 "config_space": space, "error": ""},
         result=_finished_result())
    experiment.refresh_from_db()

    assert experiment.data.config_space == space


def test_a_resumed_run_counts_only_its_own_trials(cluster_settings, experiment,
                                                   monkeypatch):
    """`offset` is the job's starting trial count."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 2, "stopped_by": "",
                          "config_space": None, "error": ""},
                  result=_finished_result(trials=5))

    assert run.trial_offset == 2
    assert run.trial_count == 3


def test_partial_results_reach_the_experiment_while_it_runs(cluster_settings,
                                                            experiment, monkeypatch):
    """The page polls `run_status` and redraws from `Experiment.result`, needing
    no cluster logic."""
    transport = FakeTransport()
    partial = _finished_result(trials=1)

    from experiments.services import run as run_service
    monkeypatch.setattr(jobs, "transport", lambda: transport)
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = jobs.workdir_for(run.id)
    transport.files[f"{workdir}/partial.json"] = json.dumps(partial).encode()
    transport.files[f"{workdir}/status.json"] = json.dumps(
        {"state": "done", "offset": 0, "stopped_by": "", "config_space": None,
         "error": ""}).encode()
    transport.files[f"{workdir}/result.json"] = json.dumps(_finished_result(3)).encode()
    transport._states = ["running", "done"]

    seen = {}
    original = run_service._finish_cluster_run

    def _capture(run_id, experiment_pk, status):
        from experiments.models import Experiment
        seen["mid_run"] = Experiment.objects.get(pk=experiment_pk).data.result
        return original(run_id, experiment_pk, status)

    monkeypatch.setattr(run_service, "_finish_cluster_run", _capture)
    run_service.execute_run(run.id)

    assert seen["mid_run"] is not None, "nothing was written while it was running"
    assert len(seen["mid_run"]["data"]) == 1


def test_a_queued_job_is_not_asked_for_results_it_cannot_have(cluster_settings,
                                                              experiment, monkeypatch):
    """`partial.json` stays unread while queued, saving ssh round trips."""
    transport = FakeTransport()
    _run(experiment, transport, monkeypatch, states=["queued", "done"],
         status={"state": "done", "offset": 0, "stopped_by": "",
                 "config_space": None, "error": ""},
         result=_finished_result())

    assert not any("partial.json" in c for c in transport.commands)


def test_cancelling_asks_rather_than_kills(cluster_settings, experiment, monkeypatch):
    """A CANCEL file checked between trials keeps paid-for trials; `scancel`
    would discard them."""
    from experiments.models import Run
    from experiments.services import run as run_service

    transport = FakeTransport()
    monkeypatch.setattr(jobs, "transport", lambda: transport)
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = jobs.workdir_for(run.id)
    Run.objects.filter(pk=run.id).update(cancel_requested=True)
    transport.files[f"{workdir}/status.json"] = json.dumps(
        {"state": "cancelled", "offset": 0, "stopped_by": "", "config_space": None,
         "error": ""}).encode()
    transport.files[f"{workdir}/result.json"] = json.dumps(_finished_result(2)).encode()
    transport._states = ["running", "done"]
    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert any("touch" in c and "CANCEL" in c for c in transport.commands)
    assert not any(c.startswith("scancel") for c in transport.commands)
    assert run.status == "cancelled"


def test_a_job_that_vanished_is_an_error_with_the_reason_it_left(cluster_settings,
                                                                 experiment, monkeypatch):
    """A job killed for time or memory leaves no status.json; stderr gives the
    reason."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch, states=["running", "failed"])

    assert run.status == "error"
    assert "exceeded memory limit" in run.error


def test_an_experiment_with_no_dataset_is_refused_before_anything_is_sent(
        cluster_settings, monkeypatch):
    """Without data the job would fail every trial, looking like a broken
    experiment."""
    from experiments.models import Experiment
    from experiments.services import run as run_service

    transport = FakeTransport()
    monkeypatch.setattr(jobs, "transport", lambda: transport)
    exp = Experiment.objects.create(
        name="dataless", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy")
    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "error"
    assert "no dataset" in run.error
    assert not transport.files, "nothing should have been sent"


def test_the_default_backend_still_runs_here(settings, experiment):
    """Local runs behave as before the cluster backend."""
    from experiments.services import run as run_service

    assert settings.RUN_BACKEND == "local"
    run = run_service.create_run(experiment, {"max_trials": 2}, "accuracy")
    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "done"
    assert run.backend == "local"
    assert run.job_id == ""


def test_a_cluster_run_refuses_a_metric_its_model_cannot_score(tmp_path, monkeypatch):
    """`run_experiment.main` refuses a probability metric for a labels-only
    model before any trial."""
    from ConfigSpace import ConfigurationSpace
    from codesigner_model import BaseModel

    from core.metrics import METRICS
    from core.optimizers import RandomOptimizer
    from remote import run_experiment

    class LabelsOnly(BaseModel):
        name = "Labels only"

        def get_config_space(self, seed: int = 0):
            return ConfigurationSpace()

        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            return list(y_train)[: len(X_val)]

    built = {"optimizer": RandomOptimizer(), "result": None, "model": LabelsOnly(), "seed": 0,
             "metrics": {n: METRICS[n] for n in ("accuracy", "roc_auc")},
             "X_train": None, "y_train": None, "X_val": None, "y_val": None, "splits": None}
    monkeypatch.setattr(run_experiment.io, "build_experiment", lambda *a, **k: ("x", built))
    (tmp_path / "snapshot.json").write_text(json.dumps({"dataset": {}, "model": {}}))
    (tmp_path / "run.json").write_text(json.dumps(
        {"primary_metric": "roc_auc", "stopping": {"max_trials": 1}}))

    run_experiment.main([str(tmp_path)])

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "error"
    assert "class probabilities" in status["error"]
