"""Handing a run to a cluster, and watching it from here.

*What:* `RUN_BACKEND=slurm` stages an experiment onto a cluster's filesystem,
submits a Slurm job, follows it, and writes back the same rows a local run
writes. The page, the run history and the export cannot tell the difference, and
that is the property worth pinning.

*How:* against a transport double returning canned `sbatch`/`squeue`/`sacct`
output. The suite must not need a cluster — or a network — so nothing here
shells out; what is tested is the conversation, which is the part that can be
wrong in ways a cluster would not reveal until a job had already run.
"""

import json

import pytest
from django.urls import reverse

from ui.services import cluster

from tests.conftest import DATASETS_DIR

pytestmark = pytest.mark.django_db


class FakeTransport:
    """A cluster that answers from a script instead of a scheduler.

    `states` is what successive polls see, so a test can walk a job through
    queued → running → done without waiting for one. Everything written is kept
    in `files`, which is also what `get` reads back — so the job's outputs are
    whatever a test puts there.
    """

    def __init__(self, states=("done",), files=None, submit="81234"):
        self.host = "fake"
        self.files: dict[str, bytes] = dict(files or {})
        self.commands: list[str] = []
        self._states = list(states)
        self._submit = submit

    # ── the two the code uses to send work ──────────────────────────────────
    def put(self, data: bytes, remote: str) -> None:
        self.files[remote] = data

    def put_file(self, local, remote: str) -> None:
        self.files[remote] = local.read_bytes()

    def get(self, remote: str):
        return self.files.get(remote)

    # ── and the one it uses to ask questions ────────────────────────────────
    def run(self, command: str):
        self.commands.append(command)
        if command.startswith("sbatch") or " sbatch " in command:
            return cluster.Command(0, self._submit + "\n", "")
        if command.startswith("squeue"):
            state = self._states[0] if len(self._states) == 1 else self._states.pop(0)
            return cluster.Command(0, {"queued": "PENDING\n", "running": "RUNNING\n"}.get(state, ""), "")
        if command.startswith("sacct"):
            state = self._states[-1]
            return cluster.Command(
                0, {"done": "COMPLETED\n", "failed": "FAILED\n", "gone": ""}.get(state, "COMPLETED\n"), "")
        if command.startswith("tail"):
            return cluster.Command(0, "slurmstepd: error: exceeded memory limit\n", "")
        return cluster.Command(0, "", "")


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
    from ui.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "remote", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 3,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def _finished_result(trials=3):
    """What a job leaves in result.json — `Experiment.result`'s own shape."""
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
    from ui.services import run as run_service

    workdir = None

    def _fake_transport():
        return transport

    monkeypatch.setattr(cluster, "transport", _fake_transport)

    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = cluster.workdir_for(run.id)
    if status is not None:
        transport.files[f"{workdir}/status.json"] = json.dumps(status).encode()
    if result is not None:
        transport.files[f"{workdir}/result.json"] = json.dumps(result).encode()
    transport._states = list(states)
    run_service.execute_run(run.id)
    run.refresh_from_db()
    return run, workdir


# ── what gets sent ───────────────────────────────────────────────────────────

def test_everything_the_job_needs_is_staged(cluster_settings, experiment, monkeypatch):
    """A snapshot, what bounds the run, and the data it was measured on.

    The dataset especially: the snapshot names it by a path that exists on this
    machine and nowhere else, so without the file beside it the job would fail
    on every trial for a reason that reads like a broken experiment.
    """
    transport = FakeTransport()
    run, workdir = _run(experiment, transport, monkeypatch,
                        status={"state": "done", "offset": 0, "stopped_by": "max_trials",
                                "config_space": None, "error": ""},
                        result=_finished_result())

    assert f"{workdir}/snapshot.json" in transport.files
    assert f"{workdir}/run.json" in transport.files
    staged = transport.files[f"{workdir}/dataset.csv"]
    assert staged == experiment.dataset.read(), "the job got a different dataset"

    sent = json.loads(transport.files[f"{workdir}/run.json"])
    assert sent["stopping"] == {"max_trials": 3}
    assert sent["primary_metric"] == "accuracy"


def test_the_job_id_is_recorded_before_the_run_is_watched(cluster_settings,
                                                          experiment, monkeypatch):
    """So a consumer that restarts mid-run can find the job again rather than
    leaving it running with nobody listening."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 0, "stopped_by": "",
                          "config_space": None, "error": ""},
                  result=_finished_result())

    assert run.job_id == "81234"
    assert run.backend == "slurm"


# ── what comes back ──────────────────────────────────────────────────────────

def test_a_finished_job_writes_the_rows_a_local_run_writes(cluster_settings,
                                                           experiment, monkeypatch):
    """Same fields, same shape — the history cannot tell where a run happened."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 0, "stopped_by": "max_trials",
                          "config_space": None, "error": ""},
                  result=_finished_result(trials=3))
    experiment.refresh_from_db()

    assert run.status == "done"
    assert run.stopped_by == "max_trials"
    assert run.trial_count == 3
    assert run.trial_seconds == pytest.approx(4.5)
    assert len(experiment.result["data"]) == 3


def test_the_search_space_the_job_saw_is_kept(cluster_settings, experiment, monkeypatch):
    """The job is the only thing that had a model to ask, exactly as a local run
    is — and every surrogate-backed figure needs the answer afterwards."""
    space = {"name": None, "hyperparameters": [], "conditions": [],
             "forbiddens": [], "python_module_version": "1.2.0",
             "format_version": 0.4}
    _run(experiment, FakeTransport(), monkeypatch,
         status={"state": "done", "offset": 0, "stopped_by": "",
                 "config_space": space, "error": ""},
         result=_finished_result())
    experiment.refresh_from_db()

    assert experiment.config_space == space


def test_a_resumed_run_counts_only_its_own_trials(cluster_settings, experiment,
                                                   monkeypatch):
    """`offset` is the job saying how many it started with."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch,
                  status={"state": "done", "offset": 2, "stopped_by": "",
                          "config_space": None, "error": ""},
                  result=_finished_result(trials=5))

    assert run.trial_offset == 2
    assert run.trial_count == 3


# ── while it is going ────────────────────────────────────────────────────────

def test_partial_results_reach_the_experiment_while_it_runs(cluster_settings,
                                                            experiment, monkeypatch):
    """Which is the whole of what the page needs: `run_status` already polls and
    the figures already redraw from `Experiment.result`, so a run on another
    machine is live without the page knowing anything about clusters."""
    transport = FakeTransport()
    partial = _finished_result(trials=1)

    from ui.services import run as run_service
    monkeypatch.setattr(cluster, "transport", lambda: transport)
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = cluster.workdir_for(run.id)
    transport.files[f"{workdir}/partial.json"] = json.dumps(partial).encode()
    transport.files[f"{workdir}/status.json"] = json.dumps(
        {"state": "done", "offset": 0, "stopped_by": "", "config_space": None,
         "error": ""}).encode()
    transport.files[f"{workdir}/result.json"] = json.dumps(_finished_result(3)).encode()
    transport._states = ["running", "done"]

    seen = {}
    original = run_service._finish_cluster_run

    def _capture(run_id, experiment_pk, status):
        from ui.models import Experiment
        seen["mid_run"] = Experiment.objects.get(pk=experiment_pk).result
        return original(run_id, experiment_pk, status)

    monkeypatch.setattr(run_service, "_finish_cluster_run", _capture)
    run_service.execute_run(run.id)

    assert seen["mid_run"] is not None, "nothing was written while it was running"
    assert len(seen["mid_run"]["data"]) == 1


def test_a_queued_job_is_not_asked_for_results_it_cannot_have(cluster_settings,
                                                              experiment, monkeypatch):
    """One ssh round trip per poll, for a file that cannot exist yet."""
    transport = FakeTransport()
    _run(experiment, transport, monkeypatch, states=["queued", "done"],
         status={"state": "done", "offset": 0, "stopped_by": "",
                 "config_space": None, "error": ""},
         result=_finished_result())

    assert not any("partial.json" in c for c in transport.commands)


# ── stopping and failing ─────────────────────────────────────────────────────

def test_cancelling_asks_rather_than_kills(cluster_settings, experiment, monkeypatch):
    """A file the run checks between trials, so the trials already paid for
    survive. `scancel` throws them away and is the escalation, not this."""
    from ui.models import Run
    from ui.services import run as run_service

    transport = FakeTransport()
    monkeypatch.setattr(cluster, "transport", lambda: transport)
    run = run_service.create_run(experiment, {"max_trials": 3}, "accuracy")
    workdir = cluster.workdir_for(run.id)
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
    """Killed for time or memory, the job never reaches its own error handling,
    so there is no status.json — and a run nobody is watching any more looks
    exactly like one still going. Its stderr is the only thing that can say."""
    run, _ = _run(experiment, FakeTransport(), monkeypatch, states=["running", "failed"])

    assert run.status == "error"
    assert "exceeded memory limit" in run.error


def test_an_experiment_with_no_dataset_is_refused_before_anything_is_sent(
        cluster_settings, monkeypatch):
    """There would be nothing for the job to measure, and it would fail every
    trial for a reason that reads like a broken experiment."""
    from ui.models import Experiment
    from ui.services import run as run_service

    transport = FakeTransport()
    monkeypatch.setattr(cluster, "transport", lambda: transport)
    exp = Experiment.objects.create(
        name="dataless", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    run = run_service.create_run(exp, {"max_trials": 3}, "accuracy")
    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "error"
    assert "no dataset" in run.error
    assert not transport.files, "nothing should have been sent"


# ── the local path is untouched ──────────────────────────────────────────────

def test_the_default_backend_still_runs_here(settings, experiment):
    """Everything that existed before this went in runs exactly as it did."""
    from ui.services import run as run_service

    assert settings.RUN_BACKEND == "local"
    run = run_service.create_run(experiment, {"max_trials": 2}, "accuracy")
    run_service.execute_run(run.id)
    run.refresh_from_db()

    assert run.status == "done"
    assert run.backend == "local"
    assert run.job_id == ""
