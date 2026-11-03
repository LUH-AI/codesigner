
from ConfigSpace import Categorical, ConfigurationSpace
from codesigner_model import OutputBaseModel

from ..forecasters import SeasonalNaiveForecaster
from .base import Tunable


class SeasonalNaive(OutputBaseModel):
    """`SeasonalNaive` is the MASE baseline forecaster."""

    name = "Seasonal Naive"
    tasks = ("regression",)
    forecaster = True
    probabilities = False
    dependencies = ("numpy",)
    #: `.save` raises `.NothingToKeep`.
    has_parameters = False

    def build(self, hyperparameters, data, seed=0):
        return SeasonalNaiveForecaster(strategy=hyperparameters["strategy"],
                                       forecast=dict(self.forecast or {"time": 0}))


class SeasonalNaiveModel(Tunable, SeasonalNaive):
    """`SeasonalNaiveModel` is a naive forecast baseline."""

    #: Series times stay unprocessed.
    takes_processing = False

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([Categorical("strategy", ["seasonal", "last", "drift"], default="seasonal")])
        return cs
