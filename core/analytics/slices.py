from __future__ import annotations

from typing import List, NamedTuple, Optional, TYPE_CHECKING

from ..metrics import metric_for
from .surrogate import from_unit, hp_grid, pair_trials_with_scores, predict_mean_std

if TYPE_CHECKING:
    from ..optimizers.base import TrialResult


#: A power of two keeps the Sobol sequence balanced, with eight samples per cell
#: of the page's 32 x 32 field.
PREDICTION_SAMPLES = 8192


class IncumbentSlice(NamedTuple):
    """`incumbent` is `(position, value)` for the one evaluated configuration on
    the slice, or None. `eta` is the best observed modelled value, a different
    number whenever `values=` is set.
    """

    grid: list
    positions: list
    mu: list
    sigma: list
    eta: Optional[float]
    incumbent: Optional[tuple]
    #: `cloud` projects predictions over the whole space onto the slice's axis,
    #: or is None without a sample.
    cloud: Optional[tuple]
    warning: Optional[str]


def sample_prediction_space(
    optimizer,
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    seed: int = 0,
    n: int = PREDICTION_SAMPLES,
    surrogate=None,
) -> dict:
    """A scrambled Sobol sample covers the space evenly on each hyperparameter's
    own scale, and the model predicts it once for every view. Projected views
    push configurations forward through the fitted PCA or PLS, so every drawn
    point is legal and lands where a trial with the same values would. `smooth`
    is False for a forest. A given *surrogate* argument reuses a model the
    caller already fitted.
    """
    import numpy as np
    from ConfigSpace import Configuration
    from ConfigSpace.util import deactivate_inactive_hyperparameters
    from scipy.stats import qmc

    from ..projection import METHODS, fit as fit_projection

    empty = {"hp_names": [], "values": [], "projections": {}, "mu": [],
             "sigma": [], "eta": None, "higher_is_better": True,
             "smooth": False, "warning": None}
    metric = metric_for(metric_name)
    scores = [t.scores[metric_name] for t in trials if metric_name in t.scores]
    if not scores:
        return dict(empty, warning=f"No trial carries a score for {metric_name}.")

    hps = list(config_space.values())
    names = [hp.name for hp in hps]
    unit = qmc.Sobol(d=len(hps), scramble=True, seed=seed).random(n)  # pyright: ignore[reportCallIssue]
    configurations, rows = [], []
    for point in unit:
        values = {hp.name: from_unit(hp, u) for hp, u in zip(hps, point)}
        try:
            configuration = Configuration(config_space, values=values)
        except Exception:  # noqa: BLE001 — a child its parent switched off
            try:
                configuration = deactivate_inactive_hyperparameters(values, config_space)
            except Exception:  # noqa: BLE001 — forbidden: not part of the space
                continue
        configurations.append(dict(configuration))
        rows.append(configuration.get_array())
    if not configurations:
        return dict(empty, warning="No configuration the space allows was sampled.")

    model, warning = surrogate, None
    if model is None:
        model, warning = optimizer.slice_surrogate(config_space, trials, metric_name, seed)
    if model is None:
        return dict(empty, warning=warning)
    try:
        mu, sigma = predict_mean_std(model, np.array(rows))
    except Exception as e:  # noqa: BLE001 — a layer must not cost the figure
        return dict(empty, warning=f"The surrogate could not predict here ({e}).")

    # Five significant figures halve the payload; the page bins coarsely anyway.
    def number(value):
        return float(f"{value:.5g}") if np.isfinite(value) else None

    projections = {}
    if len(scores) == len(trials):
        for method in METHODS:
            fitted, _reason = fit_projection(config_space, trials, scores, method)
            if fitted is None:
                continue
            projections[method] = {
                "coordinates": [[number(v) for v in row]
                                for row in fitted.transform(configurations)],
                "map": fitted.linear_map(),
                "labels": fitted.labels,
                "outlines": {str(k): fitted.outline(k)
                             for k in range(1, len(fitted.labels) + 1)},
            }

    return {
        "hp_names": names,
        "values": [[number(v) if isinstance(v, float) else v
                    for v in (c.get(name) for name in names)]
                   for c in configurations],
        # The page weights the acquisition at each sample's vectorized position,
        # where a prior's density lives.
        "vectors": [[number(v) for v in row] for row in rows],
        # Exact values, for running a picked sample as a trial.
        "configurations": configurations,
        "projections": projections,
        "mu": [number(v) for v in mu],
        "sigma": [number(v) for v in sigma],
        "eta": float(max(scores) if metric.higher_is_better else min(scores)),
        "higher_is_better": metric.higher_is_better,
        "smooth": bool(getattr(model, "smooth", False)),
        "warning": warning,
    }


def slice_axis(config_space, hp_name: str, n_points: int = 101):
    """A prior is a statement about the space, so drawing one needs no surrogate
    fit. The result is `(grid, positions, kind)`, with positions in
    ConfigSpace's normalized coordinates.
    """
    import numpy as np

    if hp_name not in config_space:
        return [], [], "continuous"
    hp = config_space[hp_name]
    grid = hp_grid(hp, n_points)
    if len(grid) < 2:
        return [], [], "continuous"

    positions = np.asarray(hp.to_vector(np.asarray(grid)), dtype=float).ravel()
    kind = "categorical" if hasattr(hp, "choices") else "continuous"
    return list(grid), [float(p) for p in positions], kind


def compute_incumbent_slice(
    optimizer,
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    hp_name: str,
    seed: int = 0,
    n_points: int = 101,
    values=None,
    surrogate=None,
    cloud=None,
) -> "IncumbentSlice":
    """A prior is a density on ConfigSpace's normalized representation, so
    `positions` gives each grid value's normalized coordinate, which is uneven
    for an integer grid and an index for a categorical. *values* models another
    quantity (such as duration), while *metric_name* still picks the incumbent
    so slices stay comparable. The caller may reuse a fitted *surrogate* and a
    *cloud* from `sample_prediction_space` across slices.
    """
    import numpy as np
    from ConfigSpace import Configuration

    if hp_name not in config_space:
        return IncumbentSlice([], [], [], [], None, None, None, f"No such hyperparameter: {hp_name}.")

    # `slice_surrogate` refits the run's own model class where the optimizer
    # can, so a Gaussian-process run reads through a Gaussian process; other
    # runs keep the forest.
    rf, warning = surrogate, None
    if rf is None:
        rf, warning = optimizer.slice_surrogate(config_space, trials, metric_name, seed,
                                           values=values)
    if rf is None:
        return IncumbentSlice([], [], [], [], None, None, None, warning)

    # The slice passes through the same scored incumbent as
    # `compute_surrogate_uncertainty`, an evaluated configuration.
    metric = metric_for(metric_name)
    scored = [t for t in trials if metric_name in t.scores]
    if not scored:
        return IncumbentSlice([], [], [], [], None, None, None, f"No trial carries a score for {metric_name}.")
    picker = max if metric.higher_is_better else min
    best = picker(scored, key=lambda t: t.scores[metric_name])

    hp = config_space[hp_name]
    grid = hp_grid(hp, n_points)
    if len(grid) < 2:
        return IncumbentSlice([], [], [], [], None, None, None, "A slice needs the hyperparameter to take more than one value.")

    # Three lists the browser indexes in lockstep drop a rejected value
    # together.
    kept_values, kept_positions, rows = [], [], []
    positions = np.asarray(hp.to_vector(np.asarray(grid)), dtype=float).ravel()
    for value, position in zip(grid, positions):
        try:
            config = dict(best.config)
            config[hp_name] = value
            rows.append(Configuration(config_space, values=config).get_array())
        except Exception:  # noqa: BLE001 — conditionals, forbidden clauses
            continue
        kept_values.append(value)
        kept_positions.append(float(position))

    if len(rows) < 2:
        return IncumbentSlice([], [], [], [], None, None, None, f"No valid configurations along {hp_name}.")

    mu, sigma = predict_mean_std(rf, np.array(rows))

    observed = [v for _, v in pair_trials_with_scores(
        config_space, trials, metric_name, values=values)]
    eta = None
    if observed:
        eta = float(picker(observed) if values is None else min(observed))

    # The incumbent is the only measured point on the line, and other trials
    # stay unmarked. The incumbent's position may fall between grid points, and
    # its value comes from `best`, since `eta` answers a different question
    # under *values*.
    incumbent = None
    try:
        position = float(np.asarray(
            hp.to_vector(np.asarray([best.config[hp_name]]))).ravel()[0])
        measured = values(best) if values is not None else best.scores.get(metric_name)
        if measured is not None:
            incumbent = (position, float(measured))
    except Exception:  # noqa: BLE001 — a marker must not cost anyone the slice
        incumbent = None

    # Only the column differs per slice, and the caller handles sampling and
    # prediction once.
    projected = None
    if cloud is not None:
        try:
            column = list(config_space.keys()).index(hp_name)
            sampled, cloud_mu, cloud_sigma = cloud
            # Inactive or unpredicted samples are left out.
            kept = [i for i in range(len(sampled))
                    if np.isfinite(sampled[i, column]) and np.isfinite(cloud_mu[i])
                    and np.isfinite(cloud_sigma[i])]
            projected = ([float(sampled[i, column]) for i in kept],
                         [float(cloud_mu[i]) for i in kept],
                         [float(cloud_sigma[i]) for i in kept])
        except Exception:  # noqa: BLE001 — a shadow must not cost the slice
            projected = None

    return IncumbentSlice(kept_values, kept_positions,
                          [float(v) for v in mu], [float(v) for v in sigma],
                          eta, incumbent, projected, None)
