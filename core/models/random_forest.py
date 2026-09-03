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

    def _fitted(self, config, X_train, y_train, seed: int):
        clf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=float(config["min_samples_split"]),
            max_features=float(config["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        return self._fitted(config, X_train, y_train, seed).predict(X_val)

    def fit_predict_proba(self, config, X_train, y_train, X_val, seed: int = 0):
        """A forest votes, so the probabilities come free — one fit answers both.

        `classes_` rather than the sorted training labels: it is the column order
        `predict_proba` actually used, and reading it off the estimator is the
        difference between a metric scoring the right class and one silently
        scoring a different one.
        """
        clf = self._fitted(config, X_train, y_train, seed)
        return clf.predict(X_val), clf.predict_proba(X_val), clf.classes_
