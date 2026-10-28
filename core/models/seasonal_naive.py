"""`core.models.seasonal_naive` holds the baseline forecasts."""

from ConfigSpace import Categorical, ConfigurationSpace
from codesigner_model import OutputBaseModel

from ..forecasters import SeasonalNaiveForecaster
from .base import Tunable


class SeasonalNaive(OutputBaseModel):
    """MASE measures every other forecaster against the seasonal
    `SeasonalNaive`."""

    name = "Seasonal Naive"
    tasks = ("regression",)
    forecaster = True
    probabilities = False
    dependencies = ("numpy",)
    #: The forecast replays history, leaving no parameters.
    has_parameters = False

    def build(self, hyperparameters, data, seed=0):
        return SeasonalNaiveForecaster(strategy=hyperparameters["strategy"],
                                       forecast=dict(self.forecast or {"time": 0}))


class SeasonalNaiveModel(Tunable, SeasonalNaive):

    #: Processing must leave a series' times alone.
    takes_processing = False

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([Categorical("strategy", ["seasonal", "last", "drift"], default="seasonal")])
        return cs
