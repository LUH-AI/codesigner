"""`RemoteModel` matches the local model's interface, so an optimizer cannot
tell the two apart. One child serves a whole run, since a fresh interpreter and
numpy import cost 1-2s against a trial that usually takes under one.
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
import tempfile
import threading
import time
import weakref
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import protocol
from ..jsonable import plain
from .deadline import DEFAULT_TRIAL_TIMEOUT, as_deadline, DeadlineSpec
from .errors import ModelProcessError, ModelTrialError, TrialCancelled, TrialTimeout

HARNESS = Path(__file__).with_name("harness.py")

#: Torch alone is slow to import.
DEFAULT_START_TIMEOUT = 120.0

_POLL = 0.25          # cancellation check interval
_STOP_GRACE = 5.0
_STDERR_LINES = 200

_INLINE_ARGUMENT = 32_000

_LIVE: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _kill_stragglers() -> None:
    for host in list(_LIVE):
        try:
            host.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown, nothing to report to
            pass


class ModelProcess:
    """`ModelProcess` takes the whole launch argv, since a bare interpreter and
    `uv run` start the harness differently, and the harness flags come after the
    argv.
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
        self._argument_files: list[str] = []


    def _argument(self, value) -> str:
        """Linux caps one argument at 128 KiB, so a wide dataset's kinds list
        goes to an ``@`` file instead."""
        text = json.dumps(value)
        if len(text) < _INLINE_ARGUMENT:
            return text
        handle, path = tempfile.mkstemp(prefix="codesigner-arg-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
        self._argument_files.append(path)
        return "@" + path

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, *, seed: int = 0, describe: bool = False, task: str | None = None,
              feature_kinds=None, forecast=None) -> dict:
        """The model learns *task*, *feature_kinds* and *forecast* before
        reporting its search space, which may depend on them.
        """
        argv = [*self._launch, "--seed", str(seed)]
        if task:
            argv += ["--task", task]
        # The model reads a missing list as all numeric, the usual case for a
        # very wide dataset.
        if feature_kinds and any(kind != "numeric" for kind in feature_kinds):
            argv += ["--feature-kinds", self._argument(list(feature_kinds))]
        if forecast:
            argv += ["--forecast", self._argument(forecast)]
        if describe:
            argv.append("--describe")

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self._env, cwd=self._cwd,
            # A new session lets the kill reach a model's own workers, such as a
            # joblib pool from n_jobs=-1.
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
        if self.alive:
            try:
                self._write({"t": protocol.SHUTDOWN, "id": self._seq + 1})
                assert self._proc is not None
                self._proc.wait(timeout=_STOP_GRACE)
            except Exception:  # noqa: BLE001 — it is going to be killed anyway
                pass
        self.kill()

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        _LIVE.discard(self)
        for path in self._argument_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._argument_files = []
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


    def _write(self, message: dict) -> None:
        try:
            payload = json.dumps(message, ensure_ascii=True, allow_nan=False)
            stdin = self._proc.stdin if self._proc is not None else None
            if stdin is None:
                raise AttributeError("the model process has no stdin")
            stdin.write(payload.encode("ascii") + b"\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AttributeError) as exc:
            raise ModelProcessError(
                f"the model process is gone{self.stderr_tail()}") from exc

    def _pump_stdout(self) -> None:
        """`_pump_stdout` reads in its own thread because `readline()` cannot
        time out, and `_await` waits on the queue instead.
        """
        proc = self._proc
        try:
            assert proc is not None and proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                self._lines.put(raw)
        except (ValueError, OSError):
            pass
        finally:
            self._lines.put(None)

    def _pump_stderr(self) -> None:
        proc = self._proc
        try:
            assert proc is not None and proc.stderr is not None
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
    """`model_session` owns the lifetime, since only a context manager can
    promise shutdown on every exit path.
    """

    def __init__(self, process: ModelProcess, hello: dict, *, seed: int = 0,
                 trial_timeout: DeadlineSpec = DEFAULT_TRIAL_TIMEOUT, cancel=None,
                 restart=None):
        self._process = process
        self._hello = hello
        self._seed = seed
        self._deadline = as_deadline(trial_timeout)
        self._cancel = cancel
        #: `model_session` supplies `_restart`, and a None restart, as in tests,
        #: leaves a dead child dead.
        self._restart = restart
        self._restart_failed = False
        self._deadline_bound = False
        self._fold = 0
        self.supports_proba = protocol.CAP_PROBA in (hello.get("capabilities") or [])
        #: `evaluate_trial` reads `last_cpu_time`, because the parent's CPU
        #: clock saw none of the work.
        self.last_cpu_time: float | None = None

    @property
    def name(self) -> str:
        return self._hello["name"]

    def get_config_space(self, seed: int = 0):
        """`from_serialized_dict` pops each hyperparameter's `type`, and SMAC
        asks for the space twice, so the deepcopy is needed. `get_config_space`
        reapplies the seed that serializing loses.
        """
        from ConfigSpace import ConfigurationSpace

        space = ConfigurationSpace.from_serialized_dict(deepcopy(self._hello["config_space"]))
        space.seed(seed)
        return space

    def use_fold(self, index: int) -> None:
        """The dataset lives in the child, so the fold travels as an index.
        `use_fold` sets the fold before the call and keeps the user-facing
        `fit_predict` signature unchanged.
        """
        self._fold = index

    def _live_process(self) -> ModelProcess:
        """`_await` kills a child that misses its deadline, and a replacement
        keeps later trials from recording fabricated crashes. The restart is
        lazy, and after one failed restart every call fails immediately.
        """
        if self._process.alive:
            return self._process
        if self._cancel is not None and self._cancel.is_set():
            # A child spawned now would hold a cancelled run for the full
            # `start_timeout`, because `start()` cannot read the cancel flag.
            raise TrialCancelled("the run was cancelled")
        if self._restart is None or self._restart_failed:
            raise ModelProcessError(
                f"the model process is gone{self._process.stderr_tail()}")
        try:
            self._process = self._restart()
        except Exception:
            self._restart_failed = True
            raise
        return self._process

    def fit_predict_proba(self, config, X_train, y_train, X_val, seed: int = 0):
        """`fit_predict_proba` raises `ModelTrialError` when the child promised
        probabilities and sent none, before a `None` fails far from its source.
        """
        y_pred, y_proba, classes = self._request(config, seed, want_proba=True)
        if y_proba is None:
            raise ModelTrialError(
                "the model offered class probabilities and then did not send any")
        return y_pred, y_proba, classes

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        """`fit_predict` ignores the arrays, since the child holds them and
        `use_fold` picks the rows.
        """
        return self._request(config, seed)[0]

    def _request(self, config, seed: int, *, want_proba: bool = False):
        process = self._live_process()
        allowed = self._deadline_seconds(config)
        started = time.monotonic()
        message = {"t": protocol.TRIAL, "seed": seed, "fold": self._fold,
                   "config": {key: plain(value) for key, value in config.items()}}
        if want_proba:
            # An older child never sees the key, which is omitted when false.
            message["want_proba"] = True
        try:
            reply = process.request(
                message, timeout=allowed, cancel=self._cancel,
            )
        except Exception:
            # A timeout, cancel or broken pipe is no duration measurement, and
            # the policy hears so.
            self._deadline.observe(config, time.monotonic() - started, completed=False)
            raise
        elapsed = time.monotonic() - started

        if reply.get("t") == protocol.ERROR:
            self._deadline.observe(config, elapsed, completed=False)
            if reply.get("kind") == protocol.KIND_TRIAL:
                raise ModelTrialError(
                    reply.get("message") or "the model failed on this trial",
                    detail=reply.get("traceback") or "")
            raise ModelProcessError(ModelProcess._explain(reply))
        if reply.get("t") != protocol.RESULT:
            self._deadline.observe(config, elapsed, completed=False)
            raise ModelProcessError(f"expected predictions, got {reply.get('t')!r}")

        self._deadline.observe(config, elapsed, completed=True)
        self.last_cpu_time = reply.get("cpu_time")
        return reply["y_pred"], reply.get("y_proba"), reply.get("classes")

    def _deadline_seconds(self, config) -> float:
        """The space is built lazily, and only for a policy that reads it.
        """
        if self._deadline.needs_config_space and not self._deadline_bound:
            self._deadline_bound = True
            try:
                self._deadline.bind(self.get_config_space(seed=self._seed))
            except Exception:  # noqa: BLE001 — an unbound policy stays on its ceiling
                pass
        return self._deadline.seconds_for(config)


def describe(launch, *, seed: int = 0, env=None, task: str | None = None,
             start_timeout: float = DEFAULT_START_TIMEOUT) -> dict:
    """`describe` returns the greeting and checks that the model imports and
    instantiates, without a web request importing user code.
    """
    process = ModelProcess(launch, start_timeout=start_timeout, env=env)
    try:
        return process.start(seed=seed, describe=True, task=task)
    finally:
        process.close()


def write_dataset(splits, directory: Path | None = None) -> Path:
    """The directory is private (0o700), since it holds the dataset. Pickling
    stays off, so an unencoded text column fails here before a subprocess
    starts.
    """
    directory = Path(directory or tempfile.mkdtemp(prefix="codesigner-split-"))
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)

    try:
        np.save(directory / protocol.X_FILE, np.asarray(splits.X), allow_pickle=False)
    except ValueError as exc:
        raise ValueError(
            "this dataset's features are not numeric, so they cannot be sent "
            f"to a model running in its own environment: {exc}") from exc

    for index, (train_idx, val_idx) in enumerate(splits.folds):
        for part, indices in (("train", train_idx), ("val", val_idx)):
            np.save(directory / protocol.fold_file(index, part),
                    np.asarray(indices, dtype=np.int64), allow_pickle=False)
    return directory


def fold_labels_for_json(splits) -> list:
    """Labels are often strings and would need pickling as arrays, so they
    travel in the `init` message.
    """
    y = np.asarray(splits.y)
    return [[plain(value) for value in y[train_idx]]
            for train_idx, _ in splits.folds]


class model_session:
    """`model_session` kills the child on every way out: an exception, a cancel,
    or a return.
    """

    def __init__(self, launch, splits, *,
                 seed: int = 0, task: str = "classification", feature_kinds=None,
                 forecast=None, cancel=None, env=None, cwd=None,
                 trial_timeout: DeadlineSpec = DEFAULT_TRIAL_TIMEOUT,
                 start_timeout: float = DEFAULT_START_TIMEOUT):
        self._launch = launch
        self._splits = splits
        self._seed = seed
        self._task = task
        self._feature_kinds = feature_kinds
        self._forecast = forecast
        self._cancel = cancel
        self._env = env
        self._cwd = cwd
        self._trial_timeout = trial_timeout
        self._start_timeout = start_timeout
        self._process: ModelProcess | None = None
        self._arrays_dir: Path | None = None

    def _spawn(self) -> tuple[ModelProcess, dict]:
        """A replacement child needs the same greeting and INIT, and
        `_arrays_dir` lives until `__exit__`, so a restart rereads the data.
        """
        process = ModelProcess(
            self._launch, start_timeout=self._start_timeout,
            env=self._env, cwd=self._cwd)
        hello = process.start(seed=self._seed, task=self._task,
                              feature_kinds=self._feature_kinds, forecast=self._forecast)

        reply = process.request(
            {"t": protocol.INIT,
             "arrays_dir": str(self._arrays_dir),
             "fold_labels": fold_labels_for_json(self._splits)},
            timeout=self._start_timeout, cancel=self._cancel,
        )
        if reply.get("t") != protocol.READY:
            process.kill()
            raise ModelProcessError(
                f"the model could not read the dataset: {ModelProcess._explain(reply)}")
        return process, hello

    def _restart(self) -> ModelProcess:
        self._process, _hello = self._spawn()
        return self._process

    def __enter__(self) -> RemoteModel:
        self._arrays_dir = write_dataset(self._splits)
        self._process, hello = self._spawn()
        return RemoteModel(self._process, hello, seed=self._seed,
                           trial_timeout=self._trial_timeout, cancel=self._cancel,
                           restart=self._restart)

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            self._process.close()
        if self._arrays_dir is not None:
            shutil.rmtree(self._arrays_dir, ignore_errors=True)
        return None
