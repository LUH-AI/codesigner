"""The model contracts, re-exported from the SDK, and the one a built-in adds.

The contracts live in the standalone, dependency-free ``codesigner_model``
package (``model_sdk/`` in this repo) because a model file is imported in *its
own* environment, where ``core`` does not exist and cannot be reached for.
Defining the classes twice would give two unrelated ABCs and ``issubclass``
would be False across the seam — so there is exactly one of each, and this
module only preserves the import path the application has always used.

New model files should ``from codesigner_model import BaseModel``.
"""

from codesigner_model import BaseModel, ModelBase, OptimizableBaseModel, OutputBaseModel

from ..processing import Processing, resolve

__all__ = ["BaseModel", "ModelBase", "OptimizableBaseModel", "OutputBaseModel", "Tunable"]


class Tunable(OptimizableBaseModel):
    """A built-in: an output model with a search space, tuned through its own
    `fit` and `predict`.

    Mixed into a model's output class — ``class RandomForestModel(Tunable,
    RandomForest)`` — so the class that is delivered (`RandomForest`) carries
    nothing of the search, and the one that is tuned adds only
    `get_config_space`. A trial's prediction is then exactly what the
    delivered model, fitted on the same rows, would predict.
    """

    #: How the experiment's columns are processed before the model sees them
    #: (`core.processing`), as stored: every step "auto" when it says nothing.
    #: Set by Codesigner before the first trial, like `feature_kinds`.
    processing: dict = {}

    #: Whether the experiment's processing applies to this model's columns. Not
    #: a forecaster's: its columns are a series' times, and what it learns from
    #: is the series itself.
    takes_processing: bool = True

    def processing_for(self, n_columns):
        """The (unfitted) processing this model is given, for *n_columns*
        columns of `feature_kinds`."""
        kinds = tuple(self.feature_kinds) if len(self.feature_kinds) == n_columns \
            else ("numeric",) * n_columns
        plan, _ = resolve(self.processing, self)
        return Processing(plan, kinds)

    def fit(self, config, X, y, seed: int = 0, *, probabilities: bool = False, kinds=None):
        """The experiment's processing, fitted on these rows, in front of the
        model fitted on what it makes of them — one estimator, so whatever
        predicts with it processes new rows exactly as the training rows were.
        Without processing to do, the model alone."""
        from sklearn.pipeline import Pipeline

        processing = self.processing_for(X.shape[1])
        if not self.takes_processing or processing.is_identity():
            return super().fit(config, X, y, seed, probabilities=probabilities)
        processed = processing.fit_transform(X, y)
        out = processing.kinds_out()
        model = super().fit(config, processed, y, seed, probabilities=probabilities,
                            kinds=out if out is not None else ("numeric",) * processed.shape[1])
        return Pipeline([("processing", processing), ("model", model)])

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        return self.predict(self.fit(config, X_train, y_train, seed), X_val)

    def fit_predict_proba(self, config, X_train, y_train, X_val, seed: int = 0):
        """Labels and probabilities from the one fit. A model whose
        probabilities cost an extra fit says so in `build` (see the SVM)."""
        if self.task == "regression" or not self.probabilities:
            raise NotImplementedError(f"{self.name} gives no class probabilities here")
        fitted = self.fit(config, X_train, y_train, seed, probabilities=True)
        return (self.predict(fitted, X_val), self.predict_proba(fitted, X_val),
                fitted.classes_)
