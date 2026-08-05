"""The one class a Codesigner model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    # ConfigSpace is your dependency, not this package's. `from __future__ import
    # annotations` keeps this import from happening at runtime, so installing the
    # contract never drags anything into an environment that only wanted it.
    from ConfigSpace import ConfigurationSpace


class BaseModel(ABC):
    """A model Codesigner can tune.

    Three things: a display :attr:`name`, the search space you want tuned, and
    one method that trains on a split and predicts the validation features.

    Codesigner never asks you for a score. It keeps the validation *labels*
    back, calls :meth:`fit_predict`, and computes every metric itself from the
    predictions you return — so a model cannot see the answers, cannot grade
    itself, and every model is measured by exactly the same code.

    Your file runs in its own Python environment, in its own process. Say what
    it needs with a PEP 723 header at the top of the file::

        # /// script
        # requires-python = ">=3.11"
        # dependencies = ["scikit-learn", "ConfigSpace"]
        # ///

    One instance is built per run and reused for every trial in it, so keep
    :meth:`fit_predict` stateless: the same ``(config, seed)`` should give the
    same predictions whenever it is called, in whatever order.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name, shown in the interface and stored in ``.ihpo`` files.

        A plain class attribute — ``name = "My Model"`` — satisfies this, and is
        what you should use: the name is read from your file's source before
        anything runs it, so an experiment has something to be called while its
        environment is still being built.
        """

    @abstractmethod
    def get_config_space(self, seed: int = 0) -> "ConfigurationSpace":
        """Return the hyperparameter search space to tune, seeded with *seed*."""

    @abstractmethod
    def fit_predict(
        self,
        config: dict[str, Any],
        X_train, y_train,
        X_val,
        seed: int = 0,
    ) -> Sequence[Any]:
        """Train on ``(X_train, y_train)`` with *config*, then predict ``X_val``.

        Return one predicted label per row of ``X_val``, in order — a numpy
        array, a pandas Series or a plain list, holding whatever label type the
        dataset uses. Use *seed* for every source of randomness you have, so
        that repeating a trial reproduces it.
        """
