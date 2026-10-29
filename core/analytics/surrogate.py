from __future__ import annotations

from typing import List, TYPE_CHECKING

from ..metrics import metric_for

if TYPE_CHECKING:
    from ..optimizers.base import TrialResult


def pair_trials_with_scores(config_space, trials: List[TrialResult], metric_name: str,
                             values=None):
    """`pair_trials_with_scores` is the one place a trial becomes training data,
    for HyperSHAP and `fit_surrogate` alike. A trial the current space rejects
    is skipped, as is one where *values* raises. By default
    `pair_trials_with_scores` learns the *metric_name* score, and `lambda t:
    t.duration` models run time instead.
    """
    from ConfigSpace import Configuration

    if values is None:
        values = lambda t: t.scores[metric_name]

    data = []
    for t in trials:
        try:
            cfg = Configuration(config_space, values=t.config)
            data.append((cfg, values(t)))
        except Exception:
            continue
    return data


def pair_trials_with_oriented_scores(config_space, trials: List[TrialResult],
                                      metric_name: str, metric=None):
    """The game names assume bigger wins, so MAX on a raw cost would call the
    worst configuration the most tunable. Negating the training data keeps
    stored fields like `hyperparameter_mistunability` meaning the same and needs
    no bounds. `fit_surrogate` stays in the metric's units for partial
    dependence.
    """
    metric = metric or metric_for(metric_name)
    data = pair_trials_with_scores(config_space, trials, metric_name)
    if metric.higher_is_better:
        return data
    return [(cfg, -score) for cfg, score in data]


def fit_surrogate(config_space, trials: List[TrialResult], metric_name: str, seed: int = 0,
                  values=None):
    """`optimize()` discards the real model, so any later stand-in needs a fresh
    fit. The result is (surrogate, warning), with None for fewer than two usable
    trials or a failed fit. With *values*, *metric_name* only labels the
    warning.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    data = pair_trials_with_scores(config_space, trials, metric_name, values=values)
    if len(data) < 2:
        return None, "Not enough trials to fit a surrogate."

    try:
        X = np.array([cfg.get_array() for cfg, _ in data])
        y = np.array([score for _, score in data])
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X, y)
        return rf, None
    except Exception as e:
        return None, f"Surrogate fit failed ({e})."


def predict_mean_std(surrogate, rows):
    """A SMAC model answers through `predict_marginalized`, the belief the
    search consulted; a forest gives tree disagreement.
    """
    import numpy as np

    predict = getattr(surrogate, "predict_marginalized", None)
    if predict is None:
        return predict_with_spread(surrogate, rows)

    mean, var = predict(np.asarray(rows))
    return (np.asarray(mean, dtype=float).ravel(),
            np.sqrt(np.clip(np.asarray(var, dtype=float), 0.0, None)).ravel())


def predict_with_spread(rf, rows):
    """`predict_with_spread` runs every tree over all rows in one pass, since
    per-call overhead dwarfs traversal, and takes mean and spread from the same
    stack. The spread is tree disagreement rather than a posterior, and far from
    the trials every tree agrees, so the spread reports confidence where data is
    missing.
    """
    import numpy as np

    per_tree = np.stack([tree.predict(rows) for tree in rf.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def hp_grid(hp, n_points: int) -> list:
    """Categorical choices are returned whole; other grids are even on the
    hyperparameter's normalized scale through `to_value`, so a log
    hyperparameter gets a log grid. A native linspace over the SVM's log-scaled
    `C` would put 19 of 20 points where the search rarely samples.
    """
    import numpy as np
    from ConfigSpace import UniformIntegerHyperparameter

    if hasattr(hp, "choices"):
        return list(hp.choices)
    raw = hp.to_value(np.linspace(0.0, 1.0, n_points))
    if isinstance(hp, UniformIntegerHyperparameter):
        return sorted({int(v) for v in raw})
    return [float(v) for v in raw]


def from_unit(hp, u: float):
    """`from_unit` follows *hp*'s own scale: logarithmic, an even share of
    choices, or a constant's value."""
    import numpy as np

    choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
    if choices:
        return choices[min(int(u * len(choices)), len(choices) - 1)]
    if not hasattr(hp, "lower"):
        return getattr(hp, "value", getattr(hp, "default_value", None))
    value = hp.to_value(np.array([u]))[0]
    return int(value) if isinstance(hp.default_value, (int, np.integer)) else float(value)


def sample_evenly(items: list, limit: int) -> list:
    """A trial history runs from exploration to exploitation, so
    `sample_evenly` spans both ends rather than a prefix, deterministically.
    """
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]
