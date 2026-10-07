"""Seasonal naive: the forecasts every other forecaster has to beat."""

from ConfigSpace import Categorical, ConfigurationSpace

from ..forecasters import SeasonalNaiveForecaster
from .base import OutputBaseModel, Tunable


class SeasonalNaive(OutputBaseModel):
    """The forecasts to beat: last season again, the last value again, or the
    straight line from the first value to the last. MASE measures every other
    forecaster against the first of these."""

    name = "Seasonal Naive"
    tasks = ("regression",)
    forecaster = True
    probabilities = False
    dependencies = ("numpy",)
    #: Its forecast is its history replayed: there is nothing fitted to keep.
    has_parameters = False

    def build(self, hyperparameters, data, seed=0):
        return SeasonalNaiveForecaster(strategy=hyperparameters["strategy"],
                                       forecast=dict(self.forecast or {"time": 0}))


class SeasonalNaiveModel(Tunable, SeasonalNaive):
    """Seasonal naive, tuned over which of its three forecasts to make."""

    #: Its columns are a series' times, which processing must leave alone.
    takes_processing = False

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([Categorical("strategy", ["seasonal", "last", "drift"], default="seasonal")])
        return cs
