"""Histogram gradient boosting: scikit-learn's, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable
from .parts import categorical_mask

#: The most labels a column can have and still be split as categories: the
#: booster bins every feature into at most 255 values, one of them kept for
#: missing.
MAX_LABELS = 254


class HistGradientBoosting(OutputBaseModel):
    """scikit-learn's histogram gradient boosting, the LightGBM idea in
    scikit-learn: fast on large tables, splits on labels and missing values as
    they are.

    Early stopping is off, so a model is the configuration it names: it would
    carve its own validation set out of the training rows and stop at a
    different iteration on every fold.
    """

    name = "HistGradientBoosting"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn", "numpy")
    native_categories = "codes"

    def build(self, hyperparameters, data, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        shared = dict(
            learning_rate=float(hyperparameters["learning_rate"]),
            max_iter=int(hyperparameters["max_iter"]),
            max_leaf_nodes=int(hyperparameters["max_leaf_nodes"]),
            min_samples_leaf=int(hyperparameters["min_samples_leaf"]),
            categorical_features=categorical_mask(data, MAX_LABELS),
            early_stopping=False,
            random_state=seed,
        )
        if self.task == "regression":
            return HistGradientBoostingRegressor(**shared)
        weight = hyperparameters.get("class_weight", "none")
        return HistGradientBoostingClassifier(
            class_weight=None if weight == "none" else weight, **shared)


class HistGradientBoostingModel(Tunable, HistGradientBoosting):
    """Histogram gradient boosting, tuned over its learning rate, size and
    leaves, and for a classification its class weighting."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float(  "learning_rate",     (0.01, 0.5), default=0.1, log=True),
            Integer("max_iter",          (20, 500),   default=100, log=True),
            Integer("max_leaf_nodes",    (4, 128),    default=31,  log=True),
            Integer("min_samples_leaf",  (2, 100),    default=20,  log=True),
        ])
        if self.task == "classification":
            cs.add(Categorical("class_weight", ["none", "balanced"], default="none"))
        return cs
