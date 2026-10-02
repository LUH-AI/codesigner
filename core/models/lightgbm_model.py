"""LightGBM's gradient boosted trees, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable
from .parts import categorical_columns


class LightGBM(OutputBaseModel):
    """LightGBM's gradient boosted trees: leaf-wise growth on binned features,
    with labels and missing values split as they are.

    A label column is handed over as its codes and named categorical when the
    booster is fitted (`fit_args`), so it groups labels itself instead of
    ordering them by code.
    """

    name = "LightGBM"
    tasks = ("classification", "regression")
    dependencies = ("lightgbm", "scikit-learn")
    native_categories = "codes"

    def build(self, hyperparameters, data, seed=0):
        from lightgbm import LGBMClassifier, LGBMRegressor

        shared = dict(
            n_estimators=int(hyperparameters["n_estimators"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            num_leaves=int(hyperparameters["num_leaves"]),
            min_child_samples=int(hyperparameters["min_child_samples"]),
            random_state=seed,
            verbose=-1,
        )
        if self.task == "regression":
            return LGBMRegressor(**shared)
        weight = hyperparameters.get("class_weight", "none")
        return LGBMClassifier(class_weight=None if weight == "none" else weight, **shared)

    def fit_args(self, hyperparameters, data, y):
        return {"categorical_feature": categorical_columns(data) or "auto"}

    def save_parameters(self, fitted, directory):
        """LightGBM's own text format: the trees, readable, and a model it can
        train on from."""
        from pathlib import Path

        fitted.booster_.save_model(str(Path(directory) / "model.txt"))
        return ["model.txt"]

    def parameter_howto(self):
        return ("import lightgbm\n"
                "booster = lightgbm.Booster(model_file='model.txt')\n"
                "# To train further: LGBMClassifier(...).fit(X, y, init_model='model.txt')\n"
                "# (a classifier's labels are classes in trial.json, in order).")


class LightGBMModel(Tunable, LightGBM):
    """LightGBM, tuned over its size, learning rate and leaves, and for a
    classification its class weighting."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",      (20, 500),  default=100, log=True),
            Float(  "learning_rate",     (0.01, 0.3), default=0.1, log=True),
            Integer("num_leaves",        (4, 128),   default=31,  log=True),
            Integer("min_child_samples", (2, 100),   default=20,  log=True),
        ])
        if self.task == "classification":
            cs.add(Categorical("class_weight", ["none", "balanced"], default="none"))
        return cs
