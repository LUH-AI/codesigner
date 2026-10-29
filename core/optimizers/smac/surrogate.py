from typing import Optional

import numpy as np
from ConfigSpace import Configuration
from scipy.stats import norm
from smac.utils.configspace import convert_configurations_to_array

from ...metrics import from_cost

# Each confidence check makes a thousand surrogate predictions, which take
# milliseconds beside a trial.
_CONFIDENCE_SAMPLES = 1000

# A GP fitted on a handful of points looks confident only because nothing
# contradicts it.
CONFIDENCE_MIN_TRIALS = 10


class RunSurrogate:
    """The model learned cost, so `from_cost` converts the mean and leaves the
    variance, which a negation or subtraction keeps.
    """

    def __init__(self, model, metric):
        self._model = model
        self._metric = metric
        #: A GP is smooth while a forest is a staircase, and a picture may be
        #: interpolated only for a GP.
        self.smooth = "Gaussian" in type(model).__name__

    def predict_marginalized(self, rows):
        """`predict_marginalized` returns `(mean, variance)`."""
        mean, var = self._model.predict_marginalized(np.asarray(rows))
        mean = np.asarray(mean, dtype=float).ravel()
        if self._metric is not None:
            mean = np.array([from_cost(self._metric, float(m)) for m in mean])
        return mean, np.asarray(var, dtype=float).ravel()


def confidence_nothing_better(smac, config_space, incumbent_config: Optional[dict]):
    """For each sampled candidate the chance of beating the incumbent is
    `Φ((μ(inc) − μ(x)) / σ(x))`, and the confidence is one minus the best
    chance. The incumbent's predicted cost is the reference, since the
    forest predicts log-scaled costs. None means an unfitted model or a
    failed prediction, and the answer holds only under the surrogate's own
    model over a sample of the space.
    """
    selector = getattr(smac.intensifier, "_config_selector", None)
    model = getattr(selector, "_model", None)
    if model is None:
        return None

    try:
        candidates = config_space.sample_configuration(_CONFIDENCE_SAMPLES)
        if not isinstance(candidates, list):
            candidates = [candidates]
        reference, _ = model.predict(
            convert_configurations_to_array(
                [Configuration(config_space, values=incumbent_config)]),
            covariance_type="diagonal")
        # The forest raises on any covariance type except "diagonal", which
        # would disable the criterion silently.
        mean, var = model.predict(
            convert_configurations_to_array(candidates), covariance_type="diagonal")
    except Exception:  # noqa: BLE001 — an unfitted or unhappy model means "no answer"
        return None
    if var is None:
        return None

    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.sqrt(np.maximum(np.asarray(var, dtype=float).reshape(-1), 0.0))
    # The floor avoids dividing by zero for a certain candidate.
    z = (float(np.asarray(reference).reshape(-1)[0]) - mean) / np.maximum(std, 1e-12)
    best_chance = float(norm.cdf(z).max())
    return 1.0 - best_chance
