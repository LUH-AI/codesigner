"""k nearest neighbours: scikit-learn's, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Integer

from .base import OutputBaseModel, Tunable


class KNN(OutputBaseModel):
    """k nearest neighbours: predicts from the training rows closest to each
    row, by majority for a label and by mean for a number.

    Fewer training rows than neighbours asked for is a model that cannot be
    fitted; it asks for every row instead.
    """

    name = "k-Nearest Neighbors"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        shared = dict(n_neighbors=min(int(hyperparameters["n_neighbors"]), data["rows"]),
                      weights=hyperparameters["weights"], p=int(hyperparameters["p"]))
        knn = (KNeighborsRegressor(**shared) if self.task == "regression"
               else KNeighborsClassifier(**shared))
        return knn


class KNNModel(Tunable, KNN):
    """k nearest neighbours, tuned over how many, how they are weighed, and
    how distance is measured."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer(    "n_neighbors", (1, 50), default=5, log=True),
            Categorical("weights",     ["uniform", "distance"], default="uniform"),
            Categorical("p",           [1, 2], default=2),
        ])
        return cs
