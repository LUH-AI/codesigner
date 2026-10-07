"""The classes a Codesigner model implements.

Three, because a model plays up to two roles and each needs different things:

* :class:`ModelBase` — what every model is: its name, the tasks it does, and
  what it is told about the data it is given.
* :class:`OptimizableBaseModel` — a model Codesigner can *tune*: a search space,
  and one method that trains on a split and predicts the held-out rows. This is
  what a model file implements, and what ``BaseModel`` has always meant.
* :class:`OutputBaseModel` — a model Codesigner can *deliver*: built from a set
  of hyperparameters, fitted, used to predict. It is what an exported model is,
  and it carries nothing of the search, so the file a person takes away runs
  without this package.

A built-in is both: its output class is the model, and a subclass adds the
search space. An exported model ships the output class alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    # ConfigSpace is your dependency, not this package's. `from __future__ import
    # annotations` keeps this import from happening at runtime, so installing the
    # contract never drags anything into an environment that only wanted it.
    from ConfigSpace import ConfigurationSpace


class ModelBase(ABC):
    """What every Codesigner model is, whichever role it plays."""

    #: The tasks this model can be tuned for: ``"classification"`` (predict a
    #: label), ``"regression"`` (predict a number), or both. A plain literal,
    #: like :attr:`name` — it is read from the source before the file runs, to
    #: decide which experiments may use the model.
    tasks: tuple = ("classification",)

    #: The task this run is for, one of :attr:`tasks`. Set by Codesigner on the
    #: instance before the first trial; a model that supports one task can
    #: ignore it.
    task: str = "classification"

    #: What kind each column of ``X`` is, in column order: ``"numeric"`` for a
    #: quantity, ``"categorical"`` for a label's code. Set by Codesigner before
    #: the first trial. Empty means every column is numeric, which is also what
    #: a dataset of numbers gives.
    feature_kinds: tuple = ()

    #: Whether this model forecasts: predicts a series' next steps from its
    #: past, reading :attr:`forecast`. A forecaster is offered only for an
    #: experiment evaluated by backtests, for each task it supports, and only
    #: there. A plain literal, like :attr:`tasks`.
    forecaster: bool = False

    #: For a run evaluated by backtests, what the forecast is: ``{"horizon",
    #: "season", "gap", "time", "series", "known"}`` — how many steps ahead, how
    #: many steps make a season, how many are skipped after the last known
    #: value, which column of ``X`` holds each row's time as a number, which
    #: names its series (None for one series), and which columns are known
    #: ahead of time. Set by Codesigner before the search space is read. The
    #: rows of ``X_train`` and ``X_val`` come in time order: a series' history
    #: is its training rows, and ``X_val`` holds its next ``horizon`` steps.
    #: Empty for every other run.
    forecast: dict = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name, shown in the interface and stored in ``.ihpo`` files.

        A plain class attribute — ``name = "My Model"`` — satisfies this, and is
        what you should use: the name is read from your file's source before
        anything runs it, so an experiment has something to be called while its
        environment is still being built.
        """


class OptimizableBaseModel(ModelBase):
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

    A model says which kinds of problem it can be tuned for in :attr:`tasks`,
    and is told which one it is being run for in :attr:`task` before the first
    trial.
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

        ``X_train`` and ``X_val`` are float arrays, whatever the dataset's
        columns held. A missing value is NaN. A text column arrives as one code
        per label — 0, 1, 2 in the labels' sorted order, NaN for a label the
        dataset does not have — and :attr:`feature_kinds` marks it
        ``"categorical"``, since the codes are not quantities: one-hot them, or
        hand them to a learner that takes categories. A date column arrives as
        its calendar parts, each of them numeric.

        Return one prediction per row of ``X_val``, in order — a numpy array, a
        pandas Series or a plain list. For classification that is a label, of
        whatever type the dataset uses; for regression, a number. Use *seed* for every source of randomness you have, so
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



#: The name every model file imports. A model file implements the tunable
#: contract, and has said ``BaseModel`` for it since before there were others.
BaseModel = OptimizableBaseModel


class OutputBaseModel(ModelBase):
    """A model as Codesigner delivers it: hyperparameters in, a fitted model out.

    Nothing here is about searching, and nothing here prepares the data:
    :meth:`build` turns a set of hyperparameters into an unfitted estimator
    with the scikit-learn interface — ``fit``, ``predict`` and, for a
    classifier, ``predict_proba`` and ``classes_`` — for columns that have
    already been processed the way the experiment says. :meth:`fit` and
    :meth:`predict` are written once here on top of it.

    What the algorithm itself can and cannot take is declared instead —
    :attr:`accepts_missing`, :attr:`scale_sensitive`,
    :attr:`native_categories` — so the experiment's processing, left to its
    defaults, gives the model what it needs, and a person who knows better can
    say otherwise.

    What :meth:`build` may know of the data is :meth:`describe`'s summary of the
    training rows — their column kinds, how many there are, the largest label
    code per column — never the rows themselves, so the same hyperparameters
    build the same model wherever it is fitted.
    """

    #: Whether a fitted classifier gives class probabilities.
    probabilities: bool = True

    #: Whether giving them costs more than the fit itself — an extra fit, a
    #: calibration. A classifier whose probabilities come free has ROC AUC and
    #: log loss tracked with every other metric; one whose do not is spared
    #: paying that on every trial.
    costly_probabilities: bool = False

    #: What a delivered copy of this model needs installed, as pip names.
    dependencies: tuple = ()

    #: Whether the algorithm takes a missing value (NaN) as it is. One that
    #: does not has its gaps filled in first, whatever processing was asked for.
    accepts_missing: bool = True

    #: Whether the algorithm measures distances or weighs columns against each
    #: other, so that a column in the thousands would drown one in fractions.
    #: Its numbers are scaled by default.
    scale_sensitive: bool = False

    #: How the algorithm takes a label column as categories of its own:
    #: ``"codes"`` (the label's code, the column named categorical), ``"text"``
    #: (the label's code as a word), or None — it has no notion of a category,
    #: and a label column is one-hot by default.
    native_categories: str | None = None

    @abstractmethod
    def build(self, hyperparameters: dict[str, Any], data: dict[str, Any], seed: int = 0):
        """The unfitted estimator *hyperparameters* describe, for data like
        *data* (see :meth:`describe`, plus ``"probabilities"`` — see
        :meth:`fit`), every source of randomness seeded with *seed*."""

    def fit_args(self, hyperparameters: dict[str, Any], data: dict[str, Any], y) -> dict[str, Any]:
        """Keyword arguments for the estimator's ``fit`` — sample weights, the
        label columns — for data like *data* with targets *y*. None by
        default."""
        return {}

    def describe(self, X, kinds=None) -> dict[str, Any]:
        """What :meth:`build` may know of training rows *X*: the kind of each
        column (*kinds*, else :attr:`feature_kinds`; all numeric when neither
        fits), how many rows and columns there are, and the largest code in
        each label column."""
        import numpy as np

        rows, columns = np.shape(X)[0], np.shape(X)[1]
        kinds = tuple(self.feature_kinds if kinds is None else kinds)
        if len(kinds) != columns:
            kinds = ("numeric",) * columns
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            # Labels already turned into words for a learner that takes them
            # so: there are no codes left to measure.
            values = None
        codes = [float(np.nanmax(values[:, i]))
                 if values is not None and kind == "categorical" and np.isfinite(values[:, i]).any()
                 else None for i, kind in enumerate(kinds)]
        return {"kinds": kinds, "rows": int(rows), "columns": int(columns), "max_code": codes}

    def fit(self, hyperparameters: dict[str, Any], X, y, seed: int = 0, *,
            probabilities: bool = False, kinds=None):
        """The model *hyperparameters* describe, fitted on all of ``(X, y)``.

        *probabilities* says the fitted model will be asked for class
        probabilities; :meth:`build` reads it as ``data["probabilities"]``, for
        a model whose probabilities cost something its labels do not. *kinds*
        is the kind of each column of *X*, when it is not
        :attr:`feature_kinds` — after processing has changed the columns.
        """
        data = dict(self.describe(X, kinds), probabilities=bool(probabilities))
        estimator = self.build(hyperparameters, data, seed)
        estimator.fit(X, y, **self.fit_args(hyperparameters, data, y))
        return estimator

    def predict(self, fitted, X):
        """*fitted*'s prediction for each row of *X*."""
        return fitted.predict(X)

    def predict_proba(self, fitted, X):
        """*fitted*'s class probabilities for each row of *X*, columns in the
        order of ``fitted.classes_``."""
        return fitted.predict_proba(X)

    # ── its parameters ──────────────────────────────────────────────────

    #: Whether a fitted model has parameters worth keeping. Not one whose
    #: prediction is its training rows replayed (a naive forecast).
    has_parameters: bool = True

    def save_parameters(self, fitted, directory) -> list:
        """Write *fitted*'s parameters into *directory*, in its library's own
        format, and return the names of the files written.

        By default skops (``model.skops``): scikit-learn's format for a fitted
        estimator, which, unlike pickle, does not run code when loaded. A model
        whose library has a format of its own — one it can load and train
        further from — overrides this.
        """
        from pathlib import Path

        from skops.io import dump

        dump(fitted, Path(directory) / "model.skops")
        return ["model.skops"]

    def parameter_howto(self) -> str:
        """How to load what :meth:`save_parameters` wrote, and to train on from it."""
        return ("from skops.io import get_untrusted_types, load\n"
                "types = get_untrusted_types(file='model.skops')  # review these first\n"
                "model = load('model.skops', trusted=types)\n"
                "# To train further: estimators with warm_start (forests, boosting)\n"
                "# take model.set_params(warm_start=True, n_estimators=...) and fit again.")


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
OptimizableBaseModel.fit_predict_proba._is_stub = True
