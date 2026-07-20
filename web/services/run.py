"""Set up an experiment from user inputs and run the optimizer synchronously.

No persistence and no threading: the caller supplies the choices, this builds
the train/val split and runs the optimizer to completion, returning the
result. Backgrounding comes in a later step.
"""

import random
from pathlib import Path

from core.io import _load_splits

from ..registry import METRICS, MODELS, OPTIMIZERS


def resolve_seed(seed_input: int) -> int:
    """A negative seed means 'pick one at random'; otherwise use it as given."""
    return random.randint(0, 2**31 - 1) if seed_input < 0 else int(seed_input)


def run_experiment(*, model_name, optimizer_name, dataset_path, seed,
                   primary_metric, n_trials, optimizer_params=None):
    """Build the split and run the chosen optimizer over all metrics.

    Every metric in the registry is scored on each trial; *primary_metric* is
    the one the optimizer optimizes. Returns the OptimizationResult.
    """
    model = MODELS[model_name]
    optimizer = type(OPTIMIZERS[optimizer_name])(**(optimizer_params or {}))
    X_train, X_val, y_train, y_val = _load_splits(Path(dataset_path), seed)
    return optimizer.optimize(
        model, X_train, y_train, X_val, y_val,
        metrics=METRICS, primary_metric=primary_metric,
        n_trials=n_trials, previous_result=None, seed=seed, cancel_event=None,
    )
