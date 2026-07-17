from ConfigSpace import ConfigurationSpace, Integer, Float
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class RandomForestModel(BaseModel):
    """Demo model: sklearn RandomForestClassifier with a 4-parameter search space."""

    name = "Random Forest"

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",      (10,  500), default=100),
            Integer("max_depth",         (2,   50),  default=10),
            Float(  "min_samples_split", (0.01, 0.5), default=0.1),
            Float(  "max_features",      (0.1,  1.0), default=0.5),
        ])
        return cs

    def train_evaluate(self, config, X_train, y_train, X_val, y_val,
                       metrics: dict, seed: int = 0) -> dict:
        clf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=float(config["min_samples_split"]),
            max_features=float(config["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        return {name: fn(y_val, y_pred) for name, fn in metrics.items()}


# Module-level sentinel — the loader looks for this name.
MODEL = RandomForestModel()
