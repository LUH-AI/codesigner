"""Logistic regression: scikit-learn's. Classification only."""

from ConfigSpace import Categorical, ConfigurationSpace, Float

from .base import OutputBaseModel, Tunable


class LogisticRegression(OutputBaseModel):
    """Logistic regression with an L2 penalty: a linear model of the classes'
    log-odds. Its regression counterpart is Elastic Net."""

    name = "Logistic Regression"
    tasks = ("classification",)
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.linear_model import LogisticRegression as Logistic

        weight = hyperparameters.get("class_weight", "none")
        model = Logistic(C=float(hyperparameters["C"]),
                         class_weight=None if weight == "none" else weight,
                         max_iter=2000, random_state=seed)
        return model


class LogisticRegressionModel(Tunable, LogisticRegression):
    """Logistic regression, tuned over its penalty and class weighting."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float(      "C",            (1e-3, 100.0), default=1.0, log=True),
            Categorical("class_weight", ["none", "balanced"], default="none"),
        ])
        return cs
