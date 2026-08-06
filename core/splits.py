"""How a trial's data is divided, whether that is one holdout or k folds.

A single holdout is fast and is all a large dataset needs. On the small tables
people actually upload it is a noisy estimate, and a search that optimizes a
noisy estimate spends its budget chasing the split rather than the model.
Cross-validation trades k times the compute for an average that means something.

Both are expressed the same way — **a list of (train, val) index pairs** —
because everything downstream then has one shape to handle: the trial loop
iterates folds, the wire protocol sends folds, and holdout is simply the case
where there is one. The alternative, a boolean and two code paths, would have
had to be threaded through the optimizers, the subprocess protocol and the
harness twice over.

It also keeps a property worth keeping. A model in its own process is sent the
training labels for each fold and never the labels it is about to be scored on.
With one fold that is the whole guarantee: the model cannot see the answers.
With k folds every row is a validation row exactly once, so across the folds of
a single trial their union is all of them, and a model that deliberately cached
what it was sent could reconstruct the labels. It cannot do so *for the fold
being scored*, which is what stops an accidental leak, but this is a weaker
statement and should not be read as the stronger one. Closing it would mean one
process per fold — k times the memory for the whole run — and that is not a
trade worth making by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

#: Below this many folds it is not cross-validation. Stored as 0 on an
#: experiment that uses a single holdout.
MIN_FOLDS = 2


@dataclass(frozen=True)
class Splits:
    """The whole dataset, and the folds a trial is evaluated over."""

    X: Any
    y: Any
    folds: List[Tuple[np.ndarray, np.ndarray]]

    @property
    def is_cv(self) -> bool:
        return len(self.folds) > 1


def holdout(X_train, y_train, X_val, y_val) -> Splits:
    """Today's single 80/20 split, expressed as one fold.

    The arrays are already divided, so they are concatenated back and indexed —
    train first, validation second — rather than re-splitting the frame. That
    keeps this exactly the split `_load_splits` produced, including its
    stratification, instead of a second attempt at reproducing it.
    """
    n_train, n_val = len(X_train), len(X_val)
    X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
    # No dtype forced. An integer target coerced to object stops looking like
    # discrete classes to scikit-learn — `type_of_target` calls it "unknown" and
    # every classifier refuses to fit, so every trial fails at once.
    y = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    return Splits(X=X, y=y, folds=[(train_idx, val_idx)])


def cross_validation(X, y, folds: int, seed: int) -> Splits:
    """*folds*-fold cross-validation over the whole dataset.

    Stratified when the labels allow it, and a plain shuffle when they do not —
    which in practice means a continuous target, since sklearn refuses to
    stratify one. A class with fewer members than there are folds only earns a
    warning and is still stratified. Seeded, so the same experiment divides the
    same way every run.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)          # dtype preserved — see `holdout` for why

    try:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        divided = list(splitter.split(X, y))
    except ValueError:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        divided = list(splitter.split(X))

    return Splits(X=X, y=y, folds=divided)
