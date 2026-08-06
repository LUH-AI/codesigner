"""Handing the dataset to a model in another process.

Sent once, when the child starts, rather than per trial: the data is identical
for every trial of a run, and a thirty-trial run would otherwise move it thirty
times. With cross-validation that matters more, not less — the same rows would
have travelled once per fold per trial.

What crosses is the whole feature matrix and, for each fold, which rows train
and which are held back. The labels of the rows being held back never go: that
is what makes "the model cannot see the answers" a property of the design rather
than a convention. See `core.splits` for how far that still holds under
cross-validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import protocol


def write_dataset(splits, directory: Path | None = None) -> Path:
    """Write the features and fold divisions for a child to read.

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
    """Each fold's *training* labels, as JSON-safe lists.

    They travel inside the `init` message rather than as array files: they are
    commonly strings, an array of those would have to be pickled, and pickling is
    the thing being avoided. There are never more of them than there are rows,
    so the size is not the problem the features can be.
    """
    y = np.asarray(splits.y)
    return [[_plain(value) for value in y[train_idx]]
            for train_idx, _ in splits.folds]


def _plain(value):
    return value.item() if hasattr(value, "item") else value
