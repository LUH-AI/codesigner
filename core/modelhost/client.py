"""Driving a model that lives in another process.

`RemoteModel` wears the local model's interface — same `name`, same
`get_config_space`, same `fit_predict` — so an optimizer cannot tell the two
apart. That is the point: the trial loops are about search, not about where the
training happened.

One child serves a whole run. Spawning per trial would pay a fresh interpreter
and a numpy import, one to two seconds, against a trial that typically takes
under one.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import weakref
from copy import deepcopy
from pathlib import Path

from . import protocol
from .arrays import fold_labels_for_json, write_dataset
from .errors import ModelProcessError, ModelTrialError, TrialCancelled, TrialTimeout

HARNESS = Path(__file__).with_name("harness.py")

#: Generous on purpose. Measured trials run 0.07–4.7s, so this is not a
#: performance budget — it is the line past which a model is presumed wedged
#: rather than slow, and a real dataset may legitimately sit in `fit` for
#: minutes.
DEFAULT_TRIAL_TIMEOUT = 600.0
#: Interpreter start plus the model file's own imports. Torch is slow to import.
DEFAULT_START_TIMEOUT = 120.0

_POLL = 0.25          # how often a wait looks up from the pipe to check for cancellation
_STOP_GRACE = 5.0
_STDERR_LINES = 200

_LIVE: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _kill_stragglers() -> None:
    """No child outlives the process that started it, however that process ends."""
    for host in list(_LIVE):
        try:
            host.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown, nothing to report to
            pass


def launch_local(python, model_file) -> list[str]:
    """Argv that runs the harness under *python*, importing *model_file*.

    The plain case: no environment of its own, so the interpreter is named
    directly. A model with a prepared environment is launched by uv instead, and
    that argv is built by the application — `core` knows nothing about uv.
    """
    return [str(python), "-B", str(HARNESS), "--model-file", str(model_file)]


class ModelProcess:
    """One live child: start it, ask it one thing at a time, make sure it dies.

    Takes the whole argv that starts the harness, because how it gets started
    differs: a bare interpreter here, `uv run` for a model with its own
    environment. Anything after this argv is the harness's own flags.
    """

    def __init__(self, launch, *, start_timeout: float = DEFAULT_START_TIMEOUT,
                 env: dict | None = None, cwd=None):
        self._launch = [str(part) for part in launch]
        self._start_timeout = start_timeout
        self._env = env
        self._cwd = str(cwd) if cwd else None
        self._proc: subprocess.Popen | None = None
        self._seq = 0
        self._lines: queue.Queue = queue.Queue()
        self._stderr: collections.deque = collections.deque(maxlen=_STDERR_LINES)

    # ── lifetime ─────────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, *, seed: int = 0, describe: bool = False) -> dict:
        """Spawn the child and return its greeting."""
        argv = [*self._launch, "--seed", str(seed)]
        if describe:
            argv.append("--describe")

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self._env, cwd=self._cwd,
            # Its own session, so killing it takes the workers it spawned for
            # itself too — a model with n_jobs=-1 leaves a joblib pool behind
            # otherwise, and those are how a machine fills up quietly.
            start_new_session=(os.name == "posix"),
        )
        _LIVE.add(self)
        self._lines = queue.Queue()
        self._stderr = collections.deque(maxlen=_STDERR_LINES)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

        hello = self._await(time.monotonic() + self._start_timeout, cancel=None)
        if hello.get("t") == protocol.ERROR:
            self.kill()
            raise ModelProcessError(self._explain(hello))
        if hello.get("t") != protocol.HELLO:
            self.kill()
            raise ModelProcessError(f"expected a greeting from the model, got {hello.get('t')!r}")
        if hello.get("protocol") != protocol.PROTOCOL_VERSION:
            self.kill()
            raise ModelProcessError(
                f"the model harness speaks protocol {hello.get('protocol')!r}, "
                f"this application speaks {protocol.PROTOCOL_VERSION}")
        return hello

    def close(self) -> None:
        """Ask the child to leave; kill it if it will not."""
        if self.alive:
            try:
                self._write({"t": protocol.SHUTDOWN, "id": self._seq + 1})
                self._proc.wait(timeout=_STOP_GRACE)
            except Exception:  # noqa: BLE001 — it is going to be killed anyway
                pass
        self.kill()

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        _LIVE.discard(self)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        try:
            proc.wait(timeout=_STOP_GRACE)
        except subprocess.TimeoutExpired:
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass

    # ── one request, one reply ───────────────────────────────────────────────

    def request(self, message: dict, *, timeout: float, cancel=None) -> dict:
        self._seq += 1
        self._write({**message, "id": self._seq})
        reply = self._await(time.monotonic() + timeout, cancel)
        if reply.get("id") != self._seq:
            self.kill()
            raise ModelProcessError(
                f"the model answered request {reply.get('id')!r} "
                f"while {self._seq} was outstanding")
        return reply

    def stderr_tail(self) -> str:
        if not self._stderr:
            return ""
        return "\n" + "".join(self._stderr).rstrip()

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _write(self, message: dict) -> None:
        try:
            payload = json.dumps(message, ensure_ascii=True, allow_nan=False)
            self._proc.stdin.write(payload.encode("ascii") + b"\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AttributeError) as exc:
            raise ModelProcessError(
                f"the model process is gone{self.stderr_tail()}") from exc

    def _pump_stdout(self) -> None:
        """A thread does nothing but read, so a wait can have a deadline.

        `readline()` cannot be polled for a timeout or a cancel flag, so the
        reading lives here and the waiting happens on a queue.
        """
        proc = self._proc
        try:
            for raw in iter(proc.stdout.readline, b""):
                self._lines.put(raw)
        except (ValueError, OSError):
            pass
        finally:
            self._lines.put(None)      # end of stream

    def _pump_stderr(self) -> None:
        proc = self._proc
        try:
            for raw in iter(proc.stderr.readline, b""):
                self._stderr.append(raw.decode("utf-8", "replace"))
        except (ValueError, OSError):
            pass

    def _await(self, deadline: float, cancel) -> dict:
        while True:
            if cancel is not None and cancel.is_set():
                self.kill()
                raise TrialCancelled("the run was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise TrialTimeout(
                    f"the model did not answer within the time allowed{self.stderr_tail()}")
            try:
                raw = self._lines.get(timeout=min(_POLL, remaining))
            except queue.Empty:
                continue
            if raw is None:
                self.kill()
                raise ModelProcessError(
                    f"the model stopped without answering{self.stderr_tail()}")
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except ValueError:
                self.kill()
                raise ModelProcessError(
                    f"the model wrote something that is not a message: {text[:200]!r}"
                    f"{self.stderr_tail()}") from None

    @staticmethod
    def _explain(error: dict) -> str:
        detail = error.get("message") or "no detail"
        trace = error.get("traceback") or ""
        return f"{detail}\n{trace}".rstrip()


class RemoteModel:
    """A model in another process, wearing the interface of one in this process.

    Built by `model_session`, which owns its lifetime — the child must be shut
    down, and only a context manager can promise that on every path out.
    """

    def __init__(self, process: ModelProcess, hello: dict, *, seed: int = 0,
                 trial_timeout: float = DEFAULT_TRIAL_TIMEOUT, cancel=None):
        self._process = process
        self._hello = hello
        self._seed = seed
        self._trial_timeout = trial_timeout
        self._cancel = cancel
        self._fold = 0
        #: Read by `evaluate_trial`: this thread's CPU clock saw none of the work,
        #: so the child's own measurement is the only true one.
        self.last_cpu_time: float | None = None

    @property
    def name(self) -> str:
        return self._hello["name"]

    def get_config_space(self, seed: int = 0):
        """The child's search space, rebuilt here as a live object.

        Copied before decoding because `from_serialized_dict` *consumes* what it
        is given — it pops each hyperparameter's `type` — so a second call on the
        same dict raises. This is called more than once per run (SMAC wants it
        for the scenario and again for importance), so the copy is load-bearing,
        not defensive.

        Serializing also loses the seed, and `RandomOptimizer` samples from this,
        so it is re-applied rather than inherited.
        """
        from ConfigSpace import ConfigurationSpace

        space = ConfigurationSpace.from_serialized_dict(deepcopy(self._hello["config_space"]))
        space.seed(seed)
        return space

    def use_fold(self, index: int) -> None:
        """Which fold the next `fit_predict` runs on.

        A local model is handed the fold as arrays. This one cannot be: the
        dataset was sent once at startup and lives in the other process, so the
        fold travels as an index and the child does the slicing. Set before the
        call rather than passed to it, because `fit_predict` is the contract
        user models implement and it must not grow an argument for something no
        user model will ever use.
        """
        self._fold = index

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        """One round trip. The arrays are already over there.

        They are accepted and ignored so this matches the local signature
        exactly; an optimizer passes them without knowing which kind of model it
        has. Which rows to use comes from `use_fold`.
        """
        reply = self._process.request(
            {"t": protocol.TRIAL, "config": _jsonable(config), "seed": seed,
             "fold": self._fold},
            timeout=self._trial_timeout, cancel=self._cancel,
        )
        if reply.get("t") == protocol.ERROR:
            if reply.get("kind") == protocol.KIND_TRIAL:
                raise ModelTrialError(reply.get("message") or "the model failed on this trial")
            raise ModelProcessError(ModelProcess._explain(reply))
        if reply.get("t") != protocol.RESULT:
            raise ModelProcessError(f"expected predictions, got {reply.get('t')!r}")

        self.last_cpu_time = reply.get("cpu_time")
        return reply["y_pred"]


def _jsonable(config: dict) -> dict:
    """A configuration as JSON. ConfigSpace hands back numpy scalars."""
    return {key: (value.item() if hasattr(value, "item") else value)
            for key, value in config.items()}


def describe(launch, *, seed: int = 0, env=None,
             start_timeout: float = DEFAULT_START_TIMEOUT) -> dict:
    """Start a model, read its greeting, stop. Its name and search space.

    How an experiment learns what it has without a web request importing user
    code, and — because the model is imported and instantiated to answer — the
    check that it works at all.
    """
    process = ModelProcess(launch, start_timeout=start_timeout, env=env)
    try:
        return process.start(seed=seed, describe=True)
    finally:
        process.close()


class model_session:
    """A running model, for the length of one optimization.

    Starts the child, hands it the split, yields a `RemoteModel`, and kills the
    child on the way out however that happens — a raised exception, a cancelled
    run, or a plain return.
    """

    def __init__(self, launch, splits, *,
                 seed: int = 0, cancel=None, env=None, cwd=None,
                 trial_timeout: float = DEFAULT_TRIAL_TIMEOUT,
                 start_timeout: float = DEFAULT_START_TIMEOUT):
        self._launch = launch
        self._splits = splits
        self._seed = seed
        self._cancel = cancel
        self._env = env
        self._cwd = cwd
        self._trial_timeout = trial_timeout
        self._start_timeout = start_timeout
        self._process: ModelProcess | None = None
        self._arrays_dir: Path | None = None

    def __enter__(self) -> RemoteModel:
        self._arrays_dir = write_dataset(self._splits)
        self._process = ModelProcess(
            self._launch, start_timeout=self._start_timeout,
            env=self._env, cwd=self._cwd)
        hello = self._process.start(seed=self._seed)

        reply = self._process.request(
            {"t": protocol.INIT,
             "arrays_dir": str(self._arrays_dir),
             "fold_labels": fold_labels_for_json(self._splits)},
            timeout=self._start_timeout, cancel=self._cancel,
        )
        if reply.get("t") != protocol.READY:
            self._process.kill()
            raise ModelProcessError(
                f"the model could not read the dataset: {ModelProcess._explain(reply)}")

        return RemoteModel(self._process, hello, seed=self._seed,
                           trial_timeout=self._trial_timeout, cancel=self._cancel)

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            self._process.close()
        if self._arrays_dir is not None:
            shutil.rmtree(self._arrays_dir, ignore_errors=True)
        return None
