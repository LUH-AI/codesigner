"""Elastic net: scikit-learn's. Regression only."""

from ConfigSpace import ConfigurationSpace, Float

from .base import OutputBaseModel, Tunable


class ElasticNet(OutputBaseModel):
    """A linear regression with a blend of L1 and L2 penalties: L1 drops
    columns that do not help, L2 shares weight among ones that move together.
    Its classification counterpart is Logistic Regression."""

    name = "Elastic Net"
    tasks = ("regression",)
    probabilities = False
    dependencies = ("scikit-learn",)
    accepts_missing = False
    scale_sensitive = True

    def build(self, hyperparameters, data, seed=0):
        from sklearn.linear_model import ElasticNet as Net

        model = Net(alpha=float(hyperparameters["alpha"]),
                    l1_ratio=float(hyperparameters["l1_ratio"]),
                    max_iter=5000, random_state=seed)
        return model


class ElasticNetModel(Tunable, ElasticNet):
    """Elastic net, tuned over its penalty's strength and blend."""

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float("alpha",    (1e-4, 10.0), default=0.01, log=True),
            Float("l1_ratio", (0.0, 1.0),   default=0.5),
        ])
        return cs
