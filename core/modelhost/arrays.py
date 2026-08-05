"""Handing the dataset split to a model in another process.

Sent once, when the child starts, rather than per trial: the split is identical
for every trial of a run, and a thirty-trial run would otherwise move it thirty
times.

Only three of the four arrays go. `y_val` stays here — that is what makes "the
model cannot see the answers" a property of the design rather than a convention.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import protocol


def write_split(X_train, X_val, directory: Path | None = None) -> Path:
    """Write the feature arrays for a child to read. Returns their directory.

    Created private to this user: it holds the dataset, and on a shared machine
    the default temp-directory permissions would not.

    Pickling is deliberately off. A feature array of Python objects — a dataset
    with a text column that was never encoded — would need it, and that dataset
    already fails at `fit`; refusing here says so before a subprocess is spent
    on it.
    """
    directory = Path(directory or tempfile.mkdtemp(prefix="codesigner-split-"))
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)

    for name, array in ((protocol.X_TRAIN_FILE, X_train), (protocol.X_VAL_FILE, X_val)):
        try:
            np.save(directory / name, np.asarray(array), allow_pickle=False)
        except ValueError as exc:
            raise ValueError(
                "this dataset's features are not numeric, so they cannot be sent "
                f"to a model running in its own environment: {exc}") from exc
    return directory


def labels_for_json(y_train) -> list:
    """`y_train` as a JSON-safe list.

    The labels travel inside the `init` message rather than as an array file:
    they are commonly strings, an array of those would have to be pickled, and
    pickling is the thing being avoided. There are as many labels as training
    rows, so the size is never the problem the features can be.
    """
    return [value.item() if hasattr(value, "item") else value for value in np.asarray(y_train)]
