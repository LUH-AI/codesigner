"""`KNN` wraps scikit-learn's neighbours."""

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Integer
from codesigner_model import OutputBaseModel

from .base import Tunable


class KNN(OutputBaseModel):
    """`build` caps `n_neighbors` at the training row count, since more cannot
    be fitted.
    """

    name = "k-Nearest Neighbors"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        shared: dict[str, Any] = dict(
            n_neighbors=min(int(hyperparameters["n_neighbors"]), data["rows"]),
            weights=hyperparameters["weights"], p=int(hyperparameters["p"]))
        knn = (KNeighborsRegressor(**shared) if self.task == "regression"
               else KNeighborsClassifier(**shared))
        return knn


class KNNModel(Tunable, KNN):

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer(    "n_neighbors", (1, 50), default=5, log=True),
            Categorical("weights",     ["uniform", "distance"], default="uniform"),
            Categorical("p",           [1, 2], default=2),
        ])
        return cs
