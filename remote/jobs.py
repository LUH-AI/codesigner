"""`jobs` runs experiments on a Slurm cluster over ssh. A Slurm job has no route
back to the app, so the job and the app exchange files on the cluster's
filesystem. `remote/run_experiment.py` documents the file contract. The host is
a name from `~/.ssh/config`, which holds the key, the user and any ProxyJump.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from django.conf import settings

#: `GONE` means the job aged out of `sacct`, an outcome distinct from failure.
QUEUED, RUNNING, DONE, FAILED, GONE = "queued", "running", "done", "failed", "gone"

#: `squeue` reports these states before a job starts.
_PENDING_STATES = {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}
_OK_STATES = {"COMPLETED"}

#: `_RUNS` names the directory under `CLUSTER_ROOT` that holds one directory per
#: run.
_RUNS = "runs"


class ClusterError(RuntimeError):
    """The cluster was unreachable or refused."""


@dataclass(frozen=True)
class Command:

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SshTransport:
    """A test swaps `SshTransport` for a double returning canned `sbatch` and
    `squeue` output.
    """

    #: `ControlPersist` keeps one shared connection open between frequent polls.
    _OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "ControlMaster=auto", "-o", "ControlPersist=60s",
             "-o", "ControlPath=~/.ssh/codesigner-%r@%h:%p"]

    def __init__(self, host: str, timeout: float = 120.0):
        if not host:
            raise ClusterError(
                "no CLUSTER_HOST is set, so there is nowhere to send this run")
        self.host = host
        self.timeout = timeout

    def run(self, command: str) -> Command:
        """`run` executes *command* on the cluster."""
        return self._exec(["ssh", *self._OPTS, self.host, command])

    def put(self, data: bytes, remote: str) -> None:
        """`put` sends in-memory payloads through `cat`, and `put_file` copies
        files.
        """
        self.run(f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        done = self._exec(["ssh", *self._OPTS, self.host,
                           f"cat > {shlex.quote(remote)}"], stdin=data)
        if not done.ok:
            raise ClusterError(f"could not write {remote}: {done.stderr.strip()}")

    def put_file(self, local: Path, remote: str) -> None:
        self.run(f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        done = self._exec(["scp", *self._OPTS, str(local), f"{self.host}:{remote}"])
        if not done.ok:
            raise ClusterError(f"could not copy {local.name}: {done.stderr.strip()}")

    def get(self, remote: str) -> bytes | None:
        """`get` returns None until *remote* exists.
        """
        done = self._exec(["ssh", *self._OPTS, self.host,
                           f"cat {shlex.quote(remote)} 2>/dev/null"], text=False)
        return cast(bytes, done.stdout) if done.ok and done.stdout else None

    def _exec(self, argv, stdin: bytes | None = None, text: bool = True) -> Command:
        try:
            done = subprocess.run(argv, input=stdin, capture_output=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ClusterError(f"{argv[0]} timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise ClusterError(f"could not run {argv[0]}: {exc}") from exc
        decode = (lambda b: (b or b"").decode("utf-8", "replace")) if text else (lambda b: b or b"")
        return Command(done.returncode, cast(str, decode(done.stdout)),
                       (done.stderr or b"").decode("utf-8", "replace"))


def transport() -> SshTransport:
    return SshTransport(settings.CLUSTER_HOST)


def workdir_for(run_id) -> str:
    """`workdir_for` names the directory after the run to keep leftovers
    traceable.
    """
    return f"{settings.CLUSTER_ROOT}/{_RUNS}/run-{run_id}"


def stage(ssh: SshTransport, run_id, snapshot: dict, run_settings: dict,
          dataset: Path, model: Path | None = None) -> str:
    """`run_experiment.py` repoints the snapshot's `dataset.path` at the staged
    dataset.
    """
    from core import io

    workdir = workdir_for(run_id)
    ssh.run(f"mkdir -p {shlex.quote(workdir)}")
    ssh.put(io.to_bytes(snapshot), f"{workdir}/snapshot.json")
    ssh.put(io.to_bytes(run_settings), f"{workdir}/run.json")
    ssh.put_file(dataset, f"{workdir}/dataset.csv")
    if model is not None:
        ssh.put_file(model, f"{workdir}/model.py")
    return workdir


def submit(ssh: SshTransport, run_id, workdir: str, *, hours: int | None = None) -> str:
    """`submit` queues the rendered job script and returns the job ID."""
    script = render_job(
        run_id=run_id, workdir=f"~/{workdir}".replace("~/", ""), hours=hours)
    ssh.put(script.encode("utf-8"), f"{workdir}/job.sbatch")
    done = ssh.run(f"cd {shlex.quote(workdir)} && sbatch --parsable job.sbatch")
    if not done.ok or not done.stdout.strip():
        raise ClusterError(f"sbatch refused this run: {done.stderr.strip() or done.stdout.strip()}")
    # `--parsable` prints `id;cluster` on a federation.
    return done.stdout.strip().splitlines()[-1].split(";")[0]


def render_job(*, run_id, workdir: str, hours: int | None = None) -> str:
    """`render_job` fills the markers of `remote/job.sbatch`.
    """
    template = Path(__file__).with_name("job.sbatch")
    hours = max(1, min(hours or settings.CLUSTER_MAX_HOURS, settings.CLUSTER_MAX_HOURS))
    values = {
        "RUN_ID": str(run_id),
        "PARTITION": settings.CLUSTER_PARTITION,
        "TIME_LIMIT": f"{hours}:00:00",
        "CPUS": str(settings.CLUSTER_CPUS),
        # `sbatch` resolves --output before the shell runs, so the path must be
        # absolute.
        "WORKDIR": f"$HOME/{workdir}",
        "ROOT": f"$HOME/{settings.CLUSTER_ROOT}",
    }
    text = template.read_text(encoding="utf-8")
    for marker, value in values.items():
        text = text.replace(f"@@{marker}@@", value)
    if "@@" in text:
        raise ClusterError("the job template has a marker nothing filled in")
    return text


def poll(ssh: SshTransport, job_id: str) -> str:
    """`squeue` forgets a finished job, so `poll` falls back to `sacct`.
    """
    live = ssh.run(f"squeue -j {shlex.quote(job_id)} --noheader -o %T")
    state = live.stdout.strip().splitlines()[0].strip() if live.stdout.strip() else ""
    if state:
        return QUEUED if state in _PENDING_STATES else RUNNING

    past = ssh.run(f"sacct -j {shlex.quote(job_id)} --noheader -X -o State")
    finished = past.stdout.strip().splitlines()[0].strip() if past.stdout.strip() else ""
    if not finished:
        return GONE
    return DONE if finished.split()[0] in _OK_STATES else FAILED


def fetch(ssh: SshTransport, workdir: str, name: str) -> dict | None:
    """`run_experiment.py` writes outputs by atomic rename, so an unparseable
    file raises `ClusterError`.
    """
    raw = ssh.get(f"{workdir}/{name}")
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ClusterError(f"{name} on the cluster is not readable: {exc}") from exc


def tail_error(ssh: SshTransport, workdir: str, lines: int = 20) -> str:
    """A scheduler kill for time or memory skips the job's error handling,
    leaving `job.err` as the only record.
    """
    done = ssh.run(f"tail -n {int(lines)} {shlex.quote(workdir)}/job.err 2>/dev/null")
    return done.stdout.strip()


def request_cancel(ssh: SshTransport, workdir: str) -> None:
    """The optimizer checks `CANCEL` between trials and keeps finished ones;
    `kill` discards them.
    """
    ssh.run(f"touch {shlex.quote(workdir)}/CANCEL")


def kill(ssh: SshTransport, job_id: str) -> None:
    """`kill` cancels the job in Slurm, losing unwritten results."""
    ssh.run(f"scancel {shlex.quote(job_id)}")
