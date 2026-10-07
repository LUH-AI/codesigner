"""CatBoost's gradient boosting, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable
from .parts import categorical_columns


class CatBoost(OutputBaseModel):
    """CatBoost's ordered gradient boosting on symmetric trees, which encodes
    categories with target statistics instead of one-hot. Its label columns
    arrive as text, which is how it takes a category (see `core.processing`)."""

    name = "CatBoost"
    tasks = ("classification", "regression")
    dependencies = ("catboost", "pandas", "numpy")
    native_categories = "text"

    def build(self, hyperparameters, data, seed=0):
        from catboost import CatBoostClassifier, CatBoostRegressor

        categorical = categorical_columns(data)
        shared = dict(
            iterations=int(hyperparameters["iterations"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            depth=int(hyperparameters["depth"]),
            l2_leaf_reg=float(hyperparameters["l2_leaf_reg"]),
            cat_features=categorical or None,
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            # It writes a training log beside wherever it runs, by default.
            allow_writing_files=False,
        )
        if self.task == "regression":
            model = CatBoostRegressor(**shared)
        else:
            balanced = hyperparameters.get("class_weight", "none") == "balanced"
            model = CatBoostClassifier(auto_class_weights="Balanced" if balanced else None,
                                       **shared)
        return model

    def save_parameters(self, fitted, directory):
        """CatBoost's own format, which it can train on from."""
        from pathlib import Path

        fitted.save_model(str(Path(directory) / "model.cbm"))
        return ["model.cbm"]

    def parameter_howto(self):
        return ("from catboost import CatBoost\n"
                "model = CatBoost(); model.load_model('model.cbm')\n"
                "# To train further: CatBoostClassifier(...).fit(X, y, init_model='model.cbm');\n"
                "# label columns as text, as processing.json says.")

    def predict(self, fitted, X):
        """One label or number per row: CatBoost's classifier answers in a
        column."""
        import numpy as np

        return np.ravel(fitted.predict(X))


class CatBoostModel(Tunable, CatBoost):
    """CatBoost, tuned over its size, learning rate, depth and leaf
    regularization, and for a classification its class weighting."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("iterations",    (50, 500),   default=100, log=True),
            Float(  "learning_rate", (0.01, 0.3), default=0.1, log=True),
            Integer("depth",         (3, 8),      default=6),
            Float(  "l2_leaf_reg",   (1.0, 10.0), default=3.0, log=True),
        ])
        if self.task == "classification":
            cs.add(Categorical("class_weight", ["none", "balanced"], default="none"))
        return cs
