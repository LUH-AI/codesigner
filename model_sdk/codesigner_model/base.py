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

    def fit_predict_proba(
        self,
        config: dict[str, Any],
        X_train, y_train,
        X_val,
        seed: int = 0,
    ) -> tuple[Sequence[Any], Sequence[Sequence[float]], Sequence[Any]]:
        """Optional. Like :meth:`fit_predict`, but also return probabilities.

        Return ``(y_pred, y_proba, classes)``:

        * ``y_pred`` — exactly what :meth:`fit_predict` returns;
        * ``y_proba`` — one row per row of ``X_val``, one column per class,
          each row summing to 1;
        * ``classes`` — the class labels, in the column order of ``y_proba``.

        Implement this only if a metric you want needs more than a label. ROC
        AUC does: it asks how well the *ranking* separates the classes, which a
        hard label cannot answer. Codesigner asks for probabilities only when a
        metric it is about to compute needs them, and tells you why it cannot
        compute one when your model has no such method.

        **Labels and probabilities together, in one call**, because a run
        scoring both accuracy and AUC would otherwise fit your model twice per
        trial for one extra number.

        **Return ``classes`` rather than letting Codesigner infer them.** The
        scoring code never sees your model, and guessing the column order from
        the training labels is right for scikit-learn and silently wrong for
        anything that orders them differently — which would not fail, it would
        just score the wrong class.

        A model that implements both should keep them consistent, most simply
        by writing::

            def fit_predict(self, config, X_train, y_train, X_val, seed=0):
                return self.fit_predict_proba(config, X_train, y_train, X_val, seed)[0]
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide class probabilities")


#: Marks the stub above as not-an-implementation.
#:
#: The host has to tell "this model has no probabilities" from "this model tried
#: and raised", and it cannot do it by asking whether the attribute exists —
#: every model inherits one now. Nor by comparing against
#: `BaseModel.fit_predict_proba`: each experiment's environment is built once
#: and cached, so a model prepared before this method existed inherits from a
#: `BaseModel` that has no such attribute to compare to.
#:
#: A flag on the function survives both. An older SDK has no attribute and no
#: flag; this one has both; an override has the attribute and not the flag.
BaseModel.fit_predict_proba._is_stub = True
