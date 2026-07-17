from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.svm import SVC

from .base import BaseModel


class SVMModel(BaseModel):
    """Demo model: sklearn SVC with a 6-parameter search space."""

    name = "SVM Classifier"

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float(      "C",        (0.01, 100.0), default=1.0,  log=True),
            Categorical("kernel",   ["rbf", "linear", "poly", "sigmoid"], default="rbf"),
            Categorical("gamma",    ["scale", "auto"],            default="scale"),
            Integer(    "degree",   (2, 5),         default=3),
            Float(      "tol",      (1e-5, 1e-1),   default=1e-3, log=True),
            Integer(    "max_iter", (200, 5000),    default=1000),
        ])
        return cs

    def train_evaluate(self, config, X_train, y_train, X_val, y_val,
                       metrics: dict, seed: int = 0) -> dict:
        clf = SVC(
            C=float(config["C"]),
            kernel=config["kernel"],
            gamma=config["gamma"],
            degree=int(config["degree"]),
            tol=float(config["tol"]),
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        return {name: fn(y_val, y_pred) for name, fn in metrics.items()}


MODEL = SVMModel()
