"""Running a model in its own process, and its own Python environment.

A user's model can need anything — a library the application does not have, or a
version of one it does. So it is not imported here: it runs as a subprocess,
under an interpreter chosen for it, and talks over a pipe.

    from core.modelhost import describe, model_session

    hello = describe(python, "model.py")          # its name and search space

    with model_session(python, "model.py", X_train, y_train, X_val) as model:
        optimizer.optimize(model, ...)            # indistinguishable from a local one

`harness.py` is the other end, and is executed rather than imported — it runs in
the model's environment, where this package does not exist.
"""

from .client import (
    DEFAULT_START_TIMEOUT,
    DEFAULT_TRIAL_TIMEOUT,
    HARNESS,
    ModelProcess,
    RemoteModel,
    describe,
    launch_local,
    model_session,
)
from .errors import (
    ModelHostError,
    ModelProcessError,
    ModelTrialError,
    TrialCancelled,
    TrialTimeout,
)

__all__ = [
    "DEFAULT_START_TIMEOUT",
    "DEFAULT_TRIAL_TIMEOUT",
    "HARNESS",
    "ModelHostError",
    "ModelProcess",
    "ModelProcessError",
    "ModelTrialError",
    "RemoteModel",
    "TrialCancelled",
    "TrialTimeout",
    "describe",
    "launch_local",
    "model_session",
]
