"""`harness.py` runs as a script in the model's environment, imports the model
file, sends the model's name and search space, and answers one configuration at
a time. `harness.py` may import only the standard library, numpy and the model's
needs, as `tests/core/test_harness_isolation.py` enforces.
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
from typing import Any

PROTOCOL_VERSION = 2
#: `CAP_PROBA` repeats the value in protocol.py, since `core` is absent here.
#: The isolation test compares only the version.
CAP_PROBA = "proba"


def _send(message: dict) -> None:
    assert sys.__stdout__ is not None
    sys.__stdout__.write(json.dumps(message, ensure_ascii=True, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def _fail(kind: str, message: str, request_id=None) -> None:
    _send({"t": "error", "id": request_id, "kind": kind,
           "message": message, "traceback": traceback.format_exc()})


def _apply_limits() -> None:
    """The child sets the limits itself because `preexec_fn` is unsafe in the
    threaded worker.
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

    # The OOM killer should pick a runaway model over its supervising worker.
    try:
        Path("/proc/self/oom_score_adj").write_text("800")
    except OSError:
        pass


def _load_model(model_file: Path):
    """`_load_model` puts the file's directory on sys.path first, and a model
    may import modules beside it.
    """
    sys.path.insert(0, str(model_file.parent))

    spec = importlib.util.spec_from_file_location("_codesigner_model", model_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def _capabilities(model) -> list:
    """`_capabilities` reads the SDK's `_is_stub` flag on the method, since
    every current model inherits `fit_predict_proba` and a cached older
    `BaseModel` lacks the method.
    """
    fn = getattr(model, "fit_predict_proba", None)
    if not callable(fn) or getattr(fn, "_is_stub", False):
        return []
    return [CAP_PROBA]


def _greeting(model, seed: int) -> dict:
    space = model.get_config_space(seed=seed)
    return {
        "t": "hello",
        "protocol": PROTOCOL_VERSION,
        "name": str(model.name),
        "model_class": type(model).__name__,
        "config_space": space.to_serialized_dict(),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "capabilities": _capabilities(model),
    }


def _load_dataset(arrays_dir: Path, fold_labels):
    """`_load_dataset` slices each fold here, so a trial names only an index.
    With allow_pickle off, a non-numeric feature column fails loudly.
    """
    import numpy as np

    X = np.load(arrays_dir / "X.npy", allow_pickle=False)
    folds = []
    for index, labels in enumerate(fold_labels):
        train_idx = np.load(arrays_dir / f"fold_{index}_train.npy", allow_pickle=False)
        val_idx = np.load(arrays_dir / f"fold_{index}_val.npy", allow_pickle=False)
        # scikit-learn refuses object-typed integer labels, so the dtype is
        # inferred.
        folds.append((X[train_idx], np.asarray(labels), X[val_idx]))
    return folds


def _argument(value: str):
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one Codesigner model.")
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--describe", action="store_true",
                        help="print the model's name and search space, then exit")
    parser.add_argument("--task", default="classification",
                        help="the task the model is run for; told to it before it is asked "
                             "for its search space")
    parser.add_argument("--feature-kinds", default="",
                        help="a JSON list: what kind each column of X is")
    parser.add_argument("--forecast", default="",
                        help="a JSON object: the forecast a forecasting model is run for")
    args = parser.parse_args(argv)

    _apply_limits()

    # stdout belongs to the protocol; the parent keeps stderr for error
    # reporting.
    real_stdout, sys.stdout = sys.stdout, sys.stderr

    try:
        model = _load_model(Path(args.model_file).resolve())
        try:
            model.task = args.task
        except AttributeError:
            pass
        if args.feature_kinds:
            try:
                model.feature_kinds = tuple(_argument(args.feature_kinds))
            except AttributeError:
                pass
        if args.forecast:
            try:
                model.forecast = _argument(args.forecast)
            except AttributeError:
                pass
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
            # Probabilities can cost several plain fits, as an SVM calibrates
            # internally, so only a metric that needs them requests them.
            want_proba = bool(request.get("want_proba"))
            y_proba: Any
            classes: Any
            cpu0 = time.process_time()
            try:
                if want_proba:
                    y_pred, y_proba, classes = model.fit_predict_proba(
                        request["config"], X_train, y_train, X_val,
                        seed=request.get("seed", 0))
                else:
                    y_pred = model.fit_predict(
                        request["config"], X_train, y_train, X_val,
                        seed=request.get("seed", 0))
                    y_proba = classes = None
            except BaseException as exc:  # noqa: BLE001 — a model may fail any way it likes
                _fail("trial", f"{type(exc).__name__}: {exc}", request_id)
                continue
            cpu = time.process_time() - cpu0

            try:
                predictions = [_plain(v) for v in y_pred]
                reply = {"t": "result", "id": request_id,
                         "y_pred": predictions, "cpu_time": cpu}
                if want_proba:
                    reply["y_proba"] = [[float(p) for p in row] for row in y_proba]
                    reply["classes"] = [_plain(c) for c in classes]
            except TypeError as exc:
                _fail("trial", f"the predictions could not be sent: {exc}", request_id)
                continue

            _send(reply)
            continue

        _fail("protocol", f"unknown request {kind!r}", request_id)
        return 1

    return 0


def _plain(value):
    """numpy scalars unwrap via `.item()`; other types are refused instead of
    stringified into a label that scores as wrong.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"a prediction of type {type(value).__name__} cannot be sent as JSON")


if __name__ == "__main__":
    sys.exit(main())
