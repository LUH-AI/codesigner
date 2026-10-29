"""`hist_gradient_boosting` wraps scikit-learn's HistGradientBoosting."""

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from codesigner_model import OutputBaseModel

from .base import Tunable
from .parts import categorical_mask

#: The histogram has at most 255 bins, one reserved for missing values.
MAX_LABELS = 254


class HistGradientBoosting(OutputBaseModel):
    """`build` disables early stopping, which would stop each fold at a
    different iteration.
    """

    name = "HistGradientBoosting"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn", "numpy")
    native_categories = "codes"

    def build(self, hyperparameters, data, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        shared: dict[str, Any] = dict(
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
