"""Random forest: scikit-learn's, classifying or regressing."""

from ConfigSpace import ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable


class RandomForest(OutputBaseModel):
    """scikit-learn's random forest. Trees take a missing value as it is and
    compare numbers only with themselves, so nothing is filled in or scaled; a
    label column is one-hot, since a split on its code would group labels by
    their alphabetical order."""

    name = "Random Forest"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)

    def build(self, hyperparameters, data, seed=0):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        forest = RandomForestRegressor if self.task == "regression" else RandomForestClassifier
        model = forest(
            n_estimators=int(hyperparameters["n_estimators"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_samples_split=float(hyperparameters["min_samples_split"]),
            max_features=float(hyperparameters["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        return model


class RandomForestModel(Tunable, RandomForest):
    """The random forest, tuned over a 4-parameter space."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",      (10,  500), default=100),
            Integer("max_depth",         (2,   50),  default=10),
            Float(  "min_samples_split", (0.01, 0.5), default=0.1),
            Float(  "max_features",      (0.1,  1.0), default=0.5),
        ])
        return cs
