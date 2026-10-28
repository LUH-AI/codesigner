"""`ETS` offers Holt-Winters exponential smoothing."""

from ConfigSpace import Categorical, ConfigurationSpace
from codesigner_model import OutputBaseModel

from ..forecasters import ETSForecaster
from .base import Tunable


class ETS(OutputBaseModel):
    """`ETS` fits statsmodels once per series."""

    name = "ETS"
    tasks = ("regression",)
    forecaster = True
    probabilities = False
    dependencies = ("numpy", "statsmodels")

    def save_parameters(self, fitted, directory):
        """`save_parameters` writes each series' statsmodels parameters as JSON."""
        import json
        from pathlib import Path

        import numpy as np

        def plain(value):
            value = np.asarray(value)
            return value.tolist() if value.ndim else value.item()

        series = {key: {name: plain(value) for name, value in result.params.items()}
                  for key, result in fitted.fits_.items()}
        (Path(directory) / "series.json").write_text(json.dumps(series, indent=2))
        return ["series.json"]

    def parameter_howto(self):
        return ("import json\n"
                "params = json.load(open('series.json'))  # per series\n"
                "# Start statsmodels' ExponentialSmoothing from them:\n"
                "# ExponentialSmoothing(values, ...).fit(smoothing_level=p['smoothing_level'], ...)")

    def build(self, hyperparameters, data, seed=0):
        return ETSForecaster(trend=hyperparameters["trend"],
                             damped=hyperparameters["damped"] == "yes",
                             seasonal=hyperparameters["seasonal"],
                             forecast=dict(self.forecast or {"time": 0}))


class ETSModel(Tunable, ETS):

    #: Processing must leave a series' times alone.
    takes_processing = False

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Categorical("trend",    ["none", "add"], default="add"),
            Categorical("damped",   ["no", "yes"], default="no"),
            Categorical("seasonal", ["none", "add", "mul"], default="add"),
        ])
        return cs
