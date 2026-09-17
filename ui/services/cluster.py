"""Running an experiment on a cluster, over ssh.

The only module that knows a cluster exists. Everything above it deals in runs
and results; everything below it is `ssh`, `sbatch` and a directory of files.

**Why a directory of files rather than a connection.** A Slurm job cannot reach
back into this process — it starts minutes later on a node with no route here —
so the two sides share the one thing they both have, which is the cluster's own
filesystem. `cluster/run_experiment.py` documents the contract; this side writes
the inputs, reads the outputs, and asks Slurm what is happening in between.

**Why the system `ssh` rather than a library.** The host is a name out of
`~/.ssh/config`, so the key, the user and any ProxyJump are configured once, in
the place the rest of the system already reads — and stay out of this
application's settings. It also adds no dependency. `ControlMaster` makes the
repeated polls one connection rather than one each.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

#: A job Slurm is still going to run, is running, has finished, or has lost.
#: `GONE` is its own answer rather than an error: a job can age out of `sacct`,
#: and "I cannot tell you" is different from "it failed", though both end a run.
QUEUED, RUNNING, DONE, FAILED, GONE = "queued", "running", "done", "failed", "gone"

#: `squeue` states that mean the job has not started yet. Anything else it
#: reports is running or on its way out, and `sacct` has the final word.
_PENDING_STATES = {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}
_OK_STATES = {"COMPLETED"}

#: Where the repository was deployed to, relative to the cluster account's home.
#: `cluster/deploy.sh` puts it there; `CLUSTER_ROOT` has to agree with it.
_RUNS = "runs"


class ClusterError(RuntimeError):
    """The cluster could not be reached, or refused something."""


@dataclass(frozen=True)
class Command:
    """One finished command: what it returned and what it said."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SshTransport:
    """Commands and files, on the far side of an ssh connection.

    A class rather than four functions so a test can put something else here —
    the suite must not need a cluster, and a double returning canned `sbatch`
    and `squeue` output is the whole of what it needs to stand in.
    """

    #: Shared connection, so a run polling every few seconds opens one channel
    #: rather than one per poll. Persisting briefly past the last command keeps
    #: the next poll cheap without holding a socket open all day.
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
        """*command*, run by a shell on the cluster."""
        return self._exec(["ssh", *self._OPTS, self.host, command])

    def put(self, data: bytes, remote: str) -> None:
        """Write *data* to *remote*, creating its directory.

        Piped through `cat` rather than scp'd from a temp file: the things sent
        are a snapshot and a small JSON document, and writing them to local disk
        first only to delete them is a step with nothing to recommend it. The
        dataset goes by `put_file`, which does have a file already.
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
        """*remote*'s contents, or None when it is not there yet.

        Absent is the ordinary case — `partial.json` does not exist until the
        first trials land — so it is an answer rather than an error.
        """
        done = self._exec(["ssh", *self._OPTS, self.host,
                           f"cat {shlex.quote(remote)} 2>/dev/null"], text=False)
        return done.stdout if done.ok and done.stdout else None

    def _exec(self, argv, stdin: bytes | None = None, text: bool = True) -> Command:
        try:
            done = subprocess.run(argv, input=stdin, capture_output=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ClusterError(f"{argv[0]} timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise ClusterError(f"could not run {argv[0]}: {exc}") from exc
        decode = (lambda b: (b or b"").decode("utf-8", "replace")) if text else (lambda b: b or b"")
        return Command(done.returncode, decode(done.stdout),
                       (done.stderr or b"").decode("utf-8", "replace"))


def transport() -> SshTransport:
    return SshTransport(settings.CLUSTER_HOST)


def workdir_for(run_id) -> str:
    """Where this run's files live on the cluster.

    Under the deployed root and named by the run, so a directory left behind is
    traceable to the row that made it — and so two runs never share one.
    """
    return f"{settings.CLUSTER_ROOT}/{_RUNS}/run-{run_id}"


# ── sending the work ─────────────────────────────────────────────────────────

def stage(ssh: SshTransport, run_id, snapshot: dict, run_settings: dict,
          dataset: Path, model: Path | None = None) -> str:
    """Put everything one run needs into a directory, and say where.

    The snapshot's `dataset.path` is left exactly as it is: the far side
    repoints it at what was staged beside it, because only the far side knows
    where that ended up. See `cluster/run_experiment.py`.
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
    """Render the job script, send it, and hand it to Slurm. Returns the job id."""
    script = render_job(
        run_id=run_id, workdir=f"~/{workdir}".replace("~/", ""), hours=hours)
    ssh.put(script.encode("utf-8"), f"{workdir}/job.sbatch")
    done = ssh.run(f"cd {shlex.quote(workdir)} && sbatch --parsable job.sbatch")
    if not done.ok or not done.stdout.strip():
        raise ClusterError(f"sbatch refused this run: {done.stderr.strip() or done.stdout.strip()}")
    # `--parsable` prints the id alone, or `id;cluster` on a federation.
    return done.stdout.strip().splitlines()[-1].split(";")[0]


def render_job(*, run_id, workdir: str, hours: int | None = None) -> str:
    """`cluster/job.sbatch` with its markers filled in.

    Kept here rather than in a string so the script an operator reads is the
    script that runs — see that file on why the markers are not `str.format`.
    """
    template = Path(settings.BASE_DIR) / "cluster" / "job.sbatch"
    hours = max(1, min(hours or settings.CLUSTER_MAX_HOURS, settings.CLUSTER_MAX_HOURS))
    values = {
        "RUN_ID": str(run_id),
        "PARTITION": settings.CLUSTER_PARTITION,
        "TIME_LIMIT": f"{hours}:00:00",
        "CPUS": str(settings.CLUSTER_CPUS),
        # Absolute, because sbatch resolves --output before the shell runs and
        # a relative path would land wherever the daemon happened to be.
        "WORKDIR": f"$HOME/{workdir}",
        "ROOT": f"$HOME/{settings.CLUSTER_ROOT}",
    }
    text = template.read_text(encoding="utf-8")
    for marker, value in values.items():
        text = text.replace(f"@@{marker}@@", value)
    if "@@" in text:
        raise ClusterError("the job template has a marker nothing filled in")
    return text


# ── watching it ──────────────────────────────────────────────────────────────

def poll(ssh: SshTransport, job_id: str) -> str:
    """What Slurm currently says about *job_id*.

    `squeue` first because it answers while the job is alive and is the cheaper
    of the two; `sacct` once it is not, because `squeue` forgets a job the moment
    it ends and silence there means "finished", not "never existed".
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
    """One of the job's JSON outputs, or None if it has not written it yet.

    A half-written file is not a case to handle: the far side writes through a
    rename, which is atomic within a directory. A file that does not parse is
    therefore a real problem and is reported as one rather than retried.
    """
    raw = ssh.get(f"{workdir}/{name}")
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ClusterError(f"{name} on the cluster is not readable: {exc}") from exc


def tail_error(ssh: SshTransport, workdir: str, lines: int = 20) -> str:
    """The end of the job's stderr, for a failure that wrote no status.

    A job killed by the scheduler — out of time, out of memory — never reaches
    its own error handling, so this is the only thing that can say why.
    """
    done = ssh.run(f"tail -n {int(lines)} {shlex.quote(workdir)}/job.err 2>/dev/null")
    return done.stdout.strip()


# ── stopping it ──────────────────────────────────────────────────────────────

def request_cancel(ssh: SshTransport, workdir: str) -> None:
    """Ask the run to stop and keep what it has.

    A file, not a signal: the optimizer checks it between trials and then writes
    its result, so the trials already paid for survive. `scancel` is the other
    thing and it throws them away — see `kill`.
    """
    ssh.run(f"touch {shlex.quote(workdir)}/CANCEL")


def kill(ssh: SshTransport, job_id: str) -> None:
    """Take the job away from Slurm, losing whatever it had not written."""
    ssh.run(f"scancel {shlex.quote(job_id)}")
