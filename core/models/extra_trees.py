"""`core.models.extra_trees` wraps scikit-learn's extra trees."""

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from codesigner_model import OutputBaseModel

from .base import Tunable


class ExtraTrees(OutputBaseModel):
    """`ExtraTrees` draws split thresholds at random, cheaper than a forest."""

    name = "Extra Trees"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)

    def build(self, hyperparameters, data, seed=0):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        shared: dict[str, Any] = dict(
            n_estimators=int(hyperparameters["n_estimators"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_samples_split=float(hyperparameters["min_samples_split"]),
            max_features=float(hyperparameters["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        if self.task == "regression":
            model = ExtraTreesRegressor(**shared)
        else:
            weight = hyperparameters.get("class_weight", "none")
            model = ExtraTreesClassifier(class_weight=None if weight == "none" else weight, **shared)
        return model


class ExtraTreesModel(Tunable, ExtraTrees):

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",      (10,  500), default=100),
            Integer("max_depth",         (2,   50),  default=20),
            Float(  "min_samples_split", (0.001, 0.5), default=0.01, log=True),
            Float(  "max_features",      (0.1,  1.0), default=0.5),
        ])
        if self.task == "classification":
            cs.add(Categorical("class_weight", ["none", "balanced"], default="none"))
        return cs
