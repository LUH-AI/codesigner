"""`ModelBase` holds what every model declares; `OptimizableBaseModel` adds a
search space and `fit_predict` for tuning, and `OutputBaseModel` builds, fits
and predicts for delivery. A built-in is both, and an exported model ships the
output class alone, so the exported file runs without `codesigner_model`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    # `from __future__ import annotations` keeps the ConfigSpace import out of
    # runtime, so the contract adds no dependency.
    from ConfigSpace import ConfigurationSpace


class ModelBase(ABC):

    #: `tasks` lists ``"classification"``, ``"regression"`` or both as a plain
    #: literal, read from source before the file runs.
    tasks: tuple = ("classification",)

    #: Set before the first trial; one of :attr:`tasks`.
    task: str = "classification"

    #: Codesigner sets one kind per column of ``X``, ``"numeric"`` or
    #: ``"categorical"``, before the first trial, and empty means all numeric.
    feature_kinds: tuple = ()

    #: A forecaster reads `forecast` and is offered only for experiments
    #: evaluated by backtests; the value must be a plain literal.
    forecaster: bool = False

    #: For backtest runs, `forecast` holds ``{"horizon", "season", "gap",
    #: "time", "series", "known"}``: steps ahead, season length, skipped steps,
    #: the time and series columns of ``X``, and the columns known ahead. Rows
    #: arrive in time order, and ``X_val`` holds each series' next ``horizon``
    #: steps; other runs get an empty dict.
    forecast: dict = {}

    if TYPE_CHECKING:
        name: str
    else:
        @property
        @abstractmethod
        def name(self) -> str:
            """A plain class attribute such as ``name = "My Model"`` satisfies
            `name`, because Codesigner reads it from source before the
            environment is built.
            """


class OptimizableBaseModel(ModelBase):
    """`OptimizableBaseModel` needs a `name`, a search space and `fit_predict`;
    Codesigner keeps the validation labels back and computes every metric from
    the returned predictions. A PEP 723 header declares the file's
    dependencies::

        # /// script
        # requires-python = ">=3.11"
        # dependencies = ["scikit-learn", "ConfigSpace"]
        # ///

    `fit_predict` must be stateless and deterministic for a given ``(config,
    seed)``, because one instance serves every trial of a run.
    """

    @abstractmethod
    def get_config_space(self, seed: int = 0) -> "ConfigurationSpace":
        """`get_config_space` seeds its search space with *seed*."""

    @abstractmethod
    def fit_predict(
        self,
        config: dict[str, Any],
        X_train, y_train,
        X_val,
        seed: int = 0,
    ) -> Sequence[Any]:
        """Inputs are float arrays with NaN for missing values; a text column
        arrives as sorted label codes that `feature_kinds` marks
        ``"categorical"``, and a date as numeric calendar parts. `fit_predict`
        returns one prediction per ``X_val`` row, a label or a number, and uses
        *seed* for all randomness.
        """

    def fit_predict_proba(
        self,
        config: dict[str, Any],
        X_train, y_train,
        X_val,
        seed: int = 0,
    ) -> tuple[Sequence[Any], Sequence[Sequence[float]], Sequence[Any]]:
        """`fit_predict_proba` is optional and returns ``(y_pred, y_proba,
        classes)``, with one probability column per class in ``classes`` order.
        Only metrics such as ROC AUC need it, and one call serves labels and
        probabilities to avoid a second fit. An inferred column order would
        score the wrong class for libraries that order differently, so the
        method returns ``classes`` explicitly. A model implementing both can
        define `fit_predict` as ``self.fit_predict_proba(...)[0]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide class probabilities")


#: Model files import this name for the tunable contract.
BaseModel = OptimizableBaseModel


class OutputBaseModel(ModelBase):
    """`OutputBaseModel.build` turns hyperparameters into an unfitted
    scikit-learn-style estimator for already processed columns, and `fit` and
    `predict` wrap it. The capability flags, such as `accepts_missing`, steer
    the default processing. `build` sees only the summary from `describe`, so
    the same hyperparameters build the same model anywhere.
    """

    #: Fitted classifiers give probabilities.
    probabilities: bool = True

    #: Probabilities needing an extra fit skip per-trial tracking.
    costly_probabilities: bool = False

    #: Pip names a delivered copy needs.
    dependencies: tuple = ()

    #: False fills gaps first, whatever processing was asked.
    accepts_missing: bool = True

    #: True scales numbers by default, for distance or weight-based algorithms.
    scale_sensitive: bool = False

    #: ``"codes"``, ``"text"``, or None, which one-hots label columns by
    #: default.
    native_categories: str | None = None

    @abstractmethod
    def build(self, hyperparameters: dict[str, Any], data: dict[str, Any], seed: int = 0) -> Any:
        """`build` returns an unfitted estimator for *data* (:meth:`describe`
        plus ``"probabilities"``) seeded with *seed*."""

    def fit_args(self, hyperparameters: dict[str, Any], data: dict[str, Any], y) -> dict[str, Any]:
        """`fit_args` supplies extra ``fit`` keyword arguments, such as sample
        weights."""
        return {}

    def describe(self, X, kinds=None) -> dict[str, Any]:
        """`describe` reports column kinds, row and column counts, and the
        largest code per label column."""
        import numpy as np

        rows, columns = np.shape(X)[0], np.shape(X)[1]
        kinds = tuple(self.feature_kinds if kinds is None else kinds)
        if len(kinds) != columns:
            kinds = ("numeric",) * columns
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            # Labels already turned into words leave no codes.
            values = None
        codes = [float(np.nanmax(values[:, i]))
                 if values is not None and kind == "categorical" and np.isfinite(values[:, i]).any()
                 else None for i, kind in enumerate(kinds)]
        return {"kinds": kinds, "rows": int(rows), "columns": int(columns), "max_code": codes}

    def fit(self, hyperparameters: dict[str, Any], X, y, seed: int = 0, *,
            probabilities: bool = False, kinds=None):
        """The *probabilities* flag reaches `build` as
        ``data["probabilities"]``, and *kinds* overrides `feature_kinds` after
        processing changed the columns.
        """
        data = dict(self.describe(X, kinds), probabilities=bool(probabilities))
        estimator = self.build(hyperparameters, data, seed)
        estimator.fit(X, y, **self.fit_args(hyperparameters, data, y))
        return estimator

    def predict(self, fitted, X):
        return fitted.predict(X)

    def predict_proba(self, fitted, X):
        return fitted.predict_proba(X)


    #: False for a replayed-history model such as a naive forecast.
    has_parameters: bool = True

    def save_parameters(self, fitted, directory) -> list:
        """The default writes skops, which loads without running code, unlike
        pickle; a library with its own format overrides `save_parameters`.
        """
        from pathlib import Path

        from skops.io import dump

        dump(fitted, Path(directory) / "model.skops")
        return ["model.skops"]

    def parameter_howto(self) -> str:
        """`parameter_howto` explains loading and further training."""
        return ("from skops.io import get_untrusted_types, load\n"
                "types = get_untrusted_types(file='model.skops')  # review these first\n"
                "model = load('model.skops', trusted=types)\n"
                "# To train further: estimators with warm_start (forests, boosting)\n"
                "# take model.set_params(warm_start=True, n_estimators=...) and fit again.")


#: The `_is_stub` flag lets the host tell a missing method from one that raised,
#: since every model inherits the attribute and an environment cached from an
#: older SDK lacks `BaseModel.fit_predict_proba`. An override carries the
#: attribute without the flag.
setattr(OptimizableBaseModel.fit_predict_proba, "_is_stub", True)
