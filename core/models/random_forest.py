
from ConfigSpace import ConfigurationSpace, Float, Integer
from codesigner_model import OutputBaseModel

from .base import Tunable


class RandomForest(OutputBaseModel):
    """`RandomForest` takes label columns one-hot, since splits on integer codes
    would group labels alphabetically."""

    name = "Random Forest"
    tasks = ("classification", "regression")
    dependencies = ("scikit-learn",)

    def build(self, hyperparameters, data, seed=0):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        forest = RandomForestRegressor if self.task == "regression" else RandomForestClassifier
        model = forest(
            n_estimators=int(hyperparameters["n_estimators"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_samples_split=float(hyperparameters["min_samples_split"]),  # pyright: ignore[reportArgumentType]
            max_features=float(hyperparameters["max_features"]),  # pyright: ignore[reportArgumentType]
            random_state=seed,
            n_jobs=-1,
        )
        return model


class RandomForestModel(Tunable, RandomForest):
    """`RandomForestModel` is a random forest needing no scaling."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",      (10,  500), default=100),
            Integer("max_depth",         (2,   50),  default=10),
            Float(  "min_samples_split", (0.01, 0.5), default=0.1),
            Float(  "max_features",      (0.1,  1.0), default=0.5),
        ])
        return cs
