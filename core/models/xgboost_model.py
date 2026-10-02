"""XGBoost's gradient boosted trees, classifying or regressing."""

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from .base import OutputBaseModel, Tunable
from .parts import LabelCoded


class XGBoost(OutputBaseModel):
    """XGBoost's gradient boosted trees, histogram method, CPU only. Labels are
    split as categories and missing values routed natively.

    It classifies the integers 0..k-1 and nothing else, so a classifier is
    wrapped to answer in the dataset's own labels (`LabelCoded`). Balanced
    class weights are sample weights (`fit_args`): XGBoost has no class weight
    of its own.
    """

    name = "XGBoost"
    tasks = ("classification", "regression")
    dependencies = ("xgboost", "scikit-learn")
    native_categories = "codes"

    def build(self, hyperparameters, data, seed=0):
        from xgboost import XGBClassifier, XGBRegressor

        kinds = data["kinds"]
        categorical = "categorical" in kinds
        shared = dict(
            n_estimators=int(hyperparameters["n_estimators"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_child_weight=float(hyperparameters["min_child_weight"]),
            tree_method="hist",
            enable_categorical=categorical,
            feature_types=["c" if kind == "categorical" else "q" for kind in kinds]
            if categorical else None,
            random_state=seed,
            n_jobs=-1,
        )
        if self.task == "regression":
            return XGBRegressor(**shared)
        return LabelCoded(XGBClassifier(**shared))

    def save_parameters(self, fitted, directory):
        """XGBoost's own JSON format: the booster, which it can train on from.
        A classifier's booster knows its labels only as 0..k-1; which label
        each is, is the `classes` beside it (see trial.json)."""
        from pathlib import Path

        booster = getattr(fitted, "classifier", fitted)
        booster.get_booster().save_model(str(Path(directory) / "model.json"))
        return ["model.json"]

    def parameter_howto(self):
        return ("import xgboost\n"
                "booster = xgboost.Booster(); booster.load_model('model.json')\n"
                "# To train further: XGBClassifier(...).fit(X, codes, xgb_model='model.json'),\n"
                "# a classifier's labels as their place among `classes` in trial.json.")

    def fit_args(self, hyperparameters, data, y):
        if self.task == "regression" or hyperparameters.get("class_weight", "none") != "balanced":
            return {}
        from sklearn.utils.class_weight import compute_sample_weight

        return {"sample_weight": compute_sample_weight("balanced", y)}


class XGBoostModel(Tunable, XGBoost):
    """XGBoost, tuned over its size, learning rate, depth and leaf weight, and
    for a classification its class weighting."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Integer("n_estimators",     (20, 500),   default=100, log=True),
            Float(  "learning_rate",    (0.01, 0.3), default=0.1, log=True),
            Integer("max_depth",        (2, 12),     default=6),
            Float(  "min_child_weight", (0.5, 20.0), default=1.0, log=True),
        ])
        if self.task == "classification":
            cs.add(Categorical("class_weight", ["none", "balanced"], default="none"))
        return cs
