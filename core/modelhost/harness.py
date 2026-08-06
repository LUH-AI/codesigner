"""Runs inside the model's own environment. Executed, never imported.

This is the other end of the pipe: it imports the user's model file, greets the
application with the model's name and search space, then answers one
configuration at a time. It is deliberately small and dependency-light — the
standard library, numpy to read the feature arrays, and whatever the model
itself needs. It must never import anything from `core` or `ui`, because neither
exists in the environment it runs in; `tests/core/test_harness_isolation.py`
enforces that.

    python harness.py --model-file /path/to/model.py
    python harness.py --model-file /path/to/model.py --describe

`--describe` prints the greeting and exits, which is how an experiment learns
what it has without running anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

PROTOCOL_VERSION = 2


def _send(message: dict) -> None:
    """One JSON object, one line, flushed.

    stdout is the protocol, so anything the model prints would corrupt it —
    which is why stdout is redirected to stderr before the model is imported.
    """
    sys.__stdout__.write(json.dumps(message, ensure_ascii=True, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def _fail(kind: str, message: str, request_id=None) -> None:
    _send({"t": "error", "id": request_id, "kind": kind,
           "message": message, "traceback": traceback.format_exc()})


def _apply_limits() -> None:
    """Limits the child sets on itself.

    Applied here rather than by the parent because `preexec_fn` is unsafe from a
    multi-threaded process, and the worker that starts us is threaded. Doing it
    after fork and before importing the model is both safe and early enough.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass

    max_bytes = os.environ.get("CODESIGNER_MAX_FILE_BYTES")
    if max_bytes:
        try:
            limit = int(max_bytes)
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        except (ValueError, OSError):
            pass

    # Prefer this process for the kernel's OOM killer, so a runaway model is
    # what dies rather than the worker that is supervising it.
    try:
        Path("/proc/self/oom_score_adj").write_text("800")
    except OSError:
        pass


def _load_model(model_file: Path):
    """Import *model_file* and instantiate the model class it defines.

    The file's own directory goes on sys.path first, so a model may import
    modules sitting beside it — a generated pipeline is more than one file more
    often than not.
    """
    sys.path.insert(0, str(model_file.parent))

    spec = importlib.util.spec_from_file_location("_codesigner_model", model_file)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a model containing dataclasses or pickling
    # anything can resolve its own module by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from codesigner_model import BaseModel

    candidates = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]
    if not candidates:
        raise RuntimeError(
            "No BaseModel subclass found in the file. Did you write "
            "`from codesigner_model import BaseModel`?")
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise RuntimeError(f"The file defines more than one model ({names}); it must define one.")
    return candidates[0]()


def _greeting(model, seed: int) -> dict:
    space = model.get_config_space(seed=seed)
    return {
        "t": "hello",
        "protocol": PROTOCOL_VERSION,
        "name": str(model.name),
        "model_class": type(model).__name__,
        "config_space": space.to_serialized_dict(),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }


def _load_dataset(arrays_dir: Path, fold_labels):
    """The feature matrix, and per fold the rows to train on and predict.

    Returns a list of `(X_train, y_train, X_val)`, one per fold, sliced here so
    a trial request only has to name an index. allow_pickle stays off: a .npy
    that needed it would be a pickle written by the parent, and keeping it off
    means a non-numeric feature column is refused loudly instead of travelling
    as one.
    """
    import numpy as np

    X = np.load(arrays_dir / "X.npy", allow_pickle=False)
    folds = []
    for index, labels in enumerate(fold_labels):
        train_idx = np.load(arrays_dir / f"fold_{index}_train.npy", allow_pickle=False)
        val_idx = np.load(arrays_dir / f"fold_{index}_val.npy", allow_pickle=False)
        folds.append((X[train_idx], np.asarray(labels, dtype=object), X[val_idx]))
    return folds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one Codesigner model.")
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--describe", action="store_true",
                        help="print the model's name and search space, then exit")
    args = parser.parse_args(argv)

    _apply_limits()

    # From here on the model may print whatever it likes; stdout belongs to the
    # protocol, so everything it writes is diverted to stderr where the parent
    # keeps it for error reporting.
    real_stdout, sys.stdout = sys.stdout, sys.stderr

    try:
        model = _load_model(Path(args.model_file).resolve())
        greeting = _greeting(model, args.seed)
    except BaseException as exc:  # noqa: BLE001 — report anything, including SystemExit
        _fail("load", f"{type(exc).__name__}: {exc}")
        return 1

    _send(greeting)
    if args.describe:
        return 0

    folds = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _fail("protocol", f"could not read a request: {exc}")
            return 1

        kind = request.get("t")
        request_id = request.get("id")

        if kind == "shutdown":
            return 0

        if kind == "init":
            try:
                folds = _load_dataset(Path(request["arrays_dir"]), request["fold_labels"])
            except BaseException as exc:  # noqa: BLE001
                _fail("load", f"could not read the dataset: {type(exc).__name__}: {exc}",
                      request_id)
                return 1
            _send({"t": "ready", "id": request_id})
            continue

        if kind == "trial":
            if folds is None:
                _fail("protocol", "a trial arrived before the dataset", request_id)
                return 1
            fold = request.get("fold", 0)
            if not 0 <= fold < len(folds):
                _fail("protocol", f"no such fold: {fold}", request_id)
                return 1
            X_train, y_train, X_val = folds[fold]
            cpu0 = time.process_time()
            try:
                y_pred = model.fit_predict(
                    request["config"], X_train, y_train, X_val, seed=request.get("seed", 0))
            except BaseException as exc:  # noqa: BLE001 — a model may fail any way it likes
                _fail("trial", f"{type(exc).__name__}: {exc}", request_id)
                continue
            cpu = time.process_time() - cpu0

            try:
                predictions = [_plain(v) for v in y_pred]
            except TypeError as exc:
                _fail("trial", f"the predictions could not be sent: {exc}", request_id)
                continue

            # Measured here because the parent's own CPU clock saw none of this.
            _send({"t": "result", "id": request_id, "y_pred": predictions, "cpu_time": cpu})
            continue

        _fail("protocol", f"unknown request {kind!r}", request_id)
        return 1

    return 0


def _plain(value):
    """A prediction as something JSON can carry.

    numpy scalars are not JSON-serializable but have `.item()`. Anything without
    that and not already a plain scalar is refused, rather than stringified into
    a label that would silently score as wrong.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"a prediction of type {type(value).__name__} cannot be sent as JSON")


if __name__ == "__main__":
    sys.exit(main())
