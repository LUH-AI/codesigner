"""`base` extends the `codesigner_model` contract."""

from typing import TYPE_CHECKING, cast

from codesigner_model import OptimizableBaseModel, OutputBaseModel

from ..processing import Processing, resolve

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class Tunable(OptimizableBaseModel, OutputBaseModel):
    """`Tunable` adds only `get_config_space` to the delivered output class, so
    a trial predicts what the delivered model would.
    """

    #: Codesigner sets `processing` before the first trial; empty means "auto".
    processing: dict = {}

    #: A forecaster's columns are times, so it skips processing.
    takes_processing: bool = True

    def processing_for(self, n_columns):
        kinds = tuple(self.feature_kinds) if len(self.feature_kinds) == n_columns \
            else ("numeric",) * n_columns
        plan, _ = resolve(self.processing, self)
        return Processing(plan, kinds)

    def fit(self, hyperparameters, X, y, seed: int = 0, *, probabilities: bool = False, kinds=None):
        """`fit` returns processing and model as one `Pipeline`, so new rows get
        the training rows' processing."""
        from sklearn.pipeline import Pipeline

        processing = self.processing_for(X.shape[1])
        if not self.takes_processing or processing.is_identity():
            return super().fit(hyperparameters, X, y, seed, probabilities=probabilities)
        processed = cast("np.ndarray | pd.DataFrame", processing.fit_transform(X, y))
        out = processing.kinds_out()
        model = super().fit(hyperparameters, processed, y, seed, probabilities=probabilities,
                            kinds=out if out is not None else ("numeric",) * processed.shape[1])
        return Pipeline([("processing", processing), ("model", model)])

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        return self.predict(self.fit(config, X_train, y_train, seed), X_val)

    def fit_predict_proba(self, config, X_train, y_train, X_val, seed: int = 0):
        """`fit_predict_proba` fits a single model with `probabilities=True`
        (see `SVM`)."""
        if self.task == "regression" or not self.probabilities:
            raise NotImplementedError(f"{self.name} gives no class probabilities here")
        fitted = self.fit(config, X_train, y_train, seed, probabilities=True)
        return (self.predict(fitted, X_val), self.predict_proba(fitted, X_val),
                fitted.classes_)
