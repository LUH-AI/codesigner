"""`core.analytics` reads a search's trials through surrogates, the HyperSHAP
games, and slices and samples of the space."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

from hypershap import ExplanationTask, HyperSHAP
from shapiq.game_theory import MoebiusConverter

from .metrics import metric_for

if TYPE_CHECKING:
    from .optimizers.base import TrialResult

#: Tunability (MAX) is the upside of tuning a hyperparameter, sensitivity (VAR)
#: is how much performance varies with it, and mistunability (MIN) is the
#: downside of a bad value. `optimizer_bias` needs an ensemble of optimizers, so
#: the app omits it.
HP_GAMES = ("tunability", "sensitivity", "mistunability")

#: A power of two keeps the Sobol sequence balanced, with eight samples per cell
#: of the page's 32 x 32 field.
PREDICTION_SAMPLES = 8192

#: 1024 coalitions admit every built-in model, the six-hyperparameter SVM
#: included (960 coalitions, 1,344 with probability metrics), and turn away a
#: 10-hyperparameter upload (12,288 coalitions, about 3 minutes). Deployments
#: override the default through `ANALYTICS_EAGER_MAX_COALITIONS`, and None means
#: no limit.
EAGER_MAX_COALITIONS = 1024


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


def _pair_trials_with_oriented_scores(config_space, trials: List[TrialResult],
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


def predict_mean_std(surrogate, rows):
    """A SMAC model answers through `predict_marginalized`, the belief the
    search consulted; a forest gives tree disagreement.
    """
    import numpy as np

    predict = getattr(surrogate, "predict_marginalized", None)
    if predict is None:
        return _predict_with_spread(surrogate, rows)

    mean, var = predict(np.asarray(rows))
    return (np.asarray(mean, dtype=float).ravel(),
            np.sqrt(np.clip(np.asarray(var, dtype=float), 0.0, None)).ravel())


def _predict_with_spread(rf, rows):
    """`_predict_with_spread` runs every tree over all rows in one pass, since
    per-call overhead dwarfs traversal, and takes mean and spread from the same
    stack. The spread is tree disagreement rather than a posterior, and far from
    the trials every tree agrees, so the spread reports confidence where data is
    missing.
    """
    import numpy as np

    per_tree = np.stack([tree.predict(rows) for tree in rf.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


class _MetricExplainer:
    """The three global games for a metric explain the same history, so
    `_MetricExplainer` builds the HyperSHAP explainer once and the fallback
    forest lazily.
    """

    __slots__ = ("hs", "warning", "insufficient", "_fallback")

    def __init__(self, hs, warning: Optional[str], insufficient: bool):
        #: None if building failed.
        self.hs = hs
        self.warning = warning
        #: Set for fewer than two trials, which get uniform weights.
        self.insufficient = insufficient
        self._fallback: Optional[tuple] = None

    def fallback_surrogate(self, config_space, trials, metric_name: str, seed: int) -> tuple:
        """`fallback_surrogate` memoizes failures too, since a failed fit fails
        the same way in every game.
        """
        if self._fallback is None:
            self._fallback = fit_surrogate(config_space, trials, metric_name, seed)
        return self._fallback


def _baseline_config(config_space, baseline: Optional[Dict[str, Any]]):
    from ConfigSpace import Configuration

    if baseline is None:
        return config_space.get_default_configuration()
    return Configuration(config_space, values=baseline)


def _moebius_terms(named: Dict[tuple, float]) -> list:
    """`_moebius_terms` returns a list of rows, because terms of every size
    would need a separator no hyperparameter name avoids. The empty coalition is
    a constant offset and is dropped.
    """
    rows = [{"members": list(members), "value": float(value)}
            for members, value in named.items() if members]
    rows.sort(key=lambda row: abs(row["value"]), reverse=True)
    return rows


def _hp_grid(hp, n_points: int) -> list:
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


def _from_unit(hp, u: float):
    """`_from_unit` follows *hp*'s own scale: logarithmic, an even share of
    choices, or a constant's value."""
    import numpy as np

    choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
    if choices:
        return choices[min(int(u * len(choices)), len(choices) - 1)]
    if not hasattr(hp, "lower"):
        return getattr(hp, "value", getattr(hp, "default_value", None))
    value = hp.to_value(np.array([u]))[0]
    return int(value) if isinstance(hp.default_value, (int, np.integer)) else float(value)


def _sample_evenly(items: list, limit: int) -> list:
    """A trial history runs from exploration to exploitation, so
    `_sample_evenly` spans both ends rather than a prefix, deterministically.
    """
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


_GameResults = tuple[Dict[str, Dict[str, float]], Dict[str, Optional[str]],
                     Dict[str, Dict[str, Dict[str, float]]], Dict[str, list], Dict[str, float]]


def eager_analytics_budget_exceeded(config_space, metrics,
                                    budget: Optional[int] = EAGER_MAX_COALITIONS) -> Optional[str]:
    """Each game evaluates 2^n_hp coalitions regardless of trial count, so a
    short run of a wide model is the expensive case. The budget counts
    coalitions, since seconds depend on the machine and a hyperparameter count
    ignores the metrics. Measured costs were 3.6s for 192 coalitions, 12.7s for
    768 and 45.6s for 3072, and the message fills the per-metric warning fields.
    """
    if not budget:
        return None
    # The list is read twice below, and a generator would leave `cost` at zero
    # and disable the guard.
    metrics = list(metrics)
    n_hp = len(list(config_space.keys()))
    cost = (2 ** n_hp) * len(HP_GAMES) * len(metrics)
    if cost <= budget:
        return None
    return (
        f"Importance analytics were skipped: {n_hp} hyperparameters over "
        f"{len(metrics)} metrics would need {cost:,} coalition "
        f"evaluations, past the {budget:,} this deployment allows "
        f"(ANALYTICS_EAGER_MAX_COALITIONS). Cost grows as 2^hyperparameters."
    )


def compute_hp_ablation(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    config_of_interest: Dict[str, Any],
    seed: int = 0,
    explainer=None,
    metric=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], Optional[str]]:
    """The ablation game explains one trial: how much each of its values helped
    or hurt against *baseline*, signed, unlike the absolute global games. A
    failure is reported rather than answered with unsigned
    `feature_importances_`. A shared *explainer* reuses one surrogate, about 35
    ms once plus 43 ms per call against 79 ms standalone.
    """
    from sklearn.ensemble import RandomForestRegressor

    if explainer is not None:
        if explainer.hs is None:
            return {}, explainer.warning or "No surrogate for a local explanation."
        return _ablate(explainer.hs, config_space, config_of_interest, baseline)

    data = _pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
    if len(data) < 2:
        return {}, "Not enough trials for a local explanation."

    try:
        task = ExplanationTask.from_data(
            config_space, data,
            base_model=RandomForestRegressor(n_estimators=100, random_state=seed))
    except Exception as e:
        return {}, f"HyperSHAP (ablation) failed ({e})."
    return _ablate(HyperSHAP(task), config_space, config_of_interest, baseline)


def _ablate(hs, config_space, config_of_interest, baseline=None):
    """`_ablate` runs the ablation on an existing HyperSHAP."""
    from ConfigSpace import Configuration

    try:
        iv = hs.ablation(
            config_of_interest=Configuration(config_space, values=config_of_interest),
            baseline_config=_baseline_config(config_space, baseline))
        named = hs.get_interaction_values_with_names(iv)
        return {members[0]: value for members, value in named.items()
                if len(members) == 1}, None
    except Exception as e:
        return {}, f"HyperSHAP (ablation) failed ({e})."


def compute_local_effects(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    seed: int = 0,
    max_trials: int = 0,
    metric=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], List[dict], Optional[str]]:
    """`compute_local_effects` runs `compute_hp_ablation` for each sampled trial
    and returns one row of per-hyperparameter effects per trial.
    `local_effects_plot` draws these rows as a beeswarm: one row of dots per
    hyperparameter and one dot per trial, so a tight cluster behaves alike
    everywhere and a fan across zero helps here and hurts there. Each row's
    `index` is its position in *trials* and `trial` its number. *max_trials*
    caps the cost, sampled by `_sample_evenly`.
    """
    explainer = _build_explainer(config_space, trials, metric_name, seed,
                                      metric=metric)
    if explainer.hs is None:
        return [], [], (explainer.warning
                        or "Not enough trials for local explanations.")

    names = list(config_space.keys())
    rows = []
    # The sample skips trials, and the page selects by position in `trials` (see
    # `_selection_meta`).
    for index, trial in _sample_evenly(list(enumerate(trials)), max_trials):
        effects, warning = compute_hp_ablation(
            config_space, trials, metric_name, trial.config, seed,
            explainer=explainer, baseline=baseline)
        if warning:
            return [], [], warning
        rows.append({"index": index, "trial": trial.trial, "effects": effects})

    if not rows:
        return [], [], "No trials to explain."
    return names, rows, None


def compute_partial_dependence(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    hp_name: str,
    seed: int = 0,
    n_points: int = 20,
    max_ice_curves: int = 0,
) -> tuple[list, List[List[Optional[float]]], List[Optional[float]], Optional[str]]:
    """Each ICE curve holds one trial's other values fixed and swaps in each
    grid value of *hp_name*; the PDP is the grid-wise mean. Averaging over
    trials weights the curve by where the search went. A rejected configuration
    leaves None, and *max_ice_curves* samples trials evenly before predicting,
    since the cost is `n_trials x n_points` predictions. All rows go through one
    `rf.predict` call, which cut 1.5s to 52ms end to end.
    """
    import numpy as np
    from ConfigSpace import Configuration

    rf, warning = fit_surrogate(config_space, trials, metric_name, seed)
    if rf is None:
        return [], [], [], warning

    grid = _hp_grid(config_space[hp_name], n_points)

    trials = _sample_evenly(trials, max_ice_curves)

    # A rejected configuration leaves a hole.
    rows, slots = [], []
    for i, t in enumerate(trials):
        for j, value in enumerate(grid):
            try:
                values = dict(t.config)
                values[hp_name] = value
                rows.append(Configuration(config_space, values=values).get_array())
            except Exception:
                continue
            slots.append((i, j))

    ice_by_trial: List[List[Optional[float]]] = [
        [None] * len(grid) for _ in trials]
    if rows:
        predictions = rf.predict(np.array(rows))
        for (i, j), value in zip(slots, predictions):
            ice_by_trial[i][j] = float(value)

    ice_lines = [row for row in ice_by_trial if any(v is not None for v in row)]

    if not ice_lines:
        return [], [], [], "No valid configurations on this hyperparameter's grid."

    pdp = []
    for i in range(len(grid)):
        column = [v for row in ice_lines if (v := row[i]) is not None]
        pdp.append(sum(column) / len(column) if column else None)

    return grid, ice_lines, pdp, None


def compute_surrogate_uncertainty(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    x_hp: str,
    y_hp: str,
    seed: int = 0,
    n_points: int = 20,
) -> tuple[list, list, List[List[Optional[float]]], Optional[str]]:
    """`z_rows` holds tree-disagreement standard deviations in the metric's
    units, None where the space rejects a pair. Tree disagreement is
    uncalibrated, and far from the trials every tree agrees, so the field turns
    confident where data is missing. The plane passes through the best trial, an
    evaluated configuration, and costs `n_points^2` rows instead of an average
    over every trial.
    """
    import numpy as np
    from ConfigSpace import Configuration

    for name in (x_hp, y_hp):
        if name not in config_space:
            return [], [], [], f"No such hyperparameter: {name}."
    if x_hp == y_hp:
        return [], [], [], "Two different hyperparameters are needed for a plane."

    rf, warning = fit_surrogate(config_space, trials, metric_name, seed)
    if rf is None:
        return [], [], [], warning

    # The incumbent comes from scored trials only, because a missing score
    # defaulting to 0.0 would win under a lower-is-better metric.
    metric = metric_for(metric_name)
    scored = [t for t in trials if metric_name in t.scores]
    if not scored:
        return [], [], [], f"No trial carries a score for {metric_name}."
    picker = max if metric.higher_is_better else min
    best = picker(scored, key=lambda t: t.scores[metric_name])
    x_grid = _hp_grid(config_space[x_hp], n_points)
    y_grid = _hp_grid(config_space[y_hp], n_points)
    if len(x_grid) < 2 or len(y_grid) < 2:
        return [], [], [], "A plane needs both axes to take more than one value."

    rows, slots = [], []
    for j, y_value in enumerate(y_grid):
        for i, x_value in enumerate(x_grid):
            try:
                values = dict(best.config)
                values[x_hp], values[y_hp] = x_value, y_value
                rows.append(Configuration(config_space, values=values).get_array())
            except Exception:  # noqa: BLE001 — conditionals, forbidden clauses
                continue
            slots.append((j, i))

    z: List[List[Optional[float]]] = [[None] * len(x_grid) for _ in y_grid]
    if not rows:
        return [], [], [], "No valid configurations on this pair of axes."

    _, spread = _predict_with_spread(rf, np.array(rows))
    for (j, i), value in zip(slots, spread):
        z[j][i] = float(value)

    return x_grid, y_grid, z, None


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

    from .projection import METHODS, fit as fit_projection

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
        values = {hp.name: _from_unit(hp, u) for hp, u in zip(hps, point)}
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
    grid = _hp_grid(hp, n_points)
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
    grid = _hp_grid(hp, n_points)
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


def _skipped_games(metrics, reason: str) -> Dict[str, tuple]:
    """The same shape as a HyperSHAP failure, and empty importance degrades
    cleanly at both consumers.
    """
    return {game: ({m: {} for m in metrics},
                   {m: reason for m in metrics},
                   {m: {} for m in metrics},
                   {m: [] for m in metrics},
                   {m: 0.0 for m in metrics})
            for game in HP_GAMES}


def compute_hp_games(
    config_space,
    trials: List[TrialResult],
    metrics,
    seed: int = 0,
    cancel_event=None,
    baseline: Optional[Dict[str, Any]] = None,
    budget: Optional[int] = EAGER_MAX_COALITIONS,
    wanted: bool = True,
) -> Dict[str, _GameResults]:
    """`compute_hp_games` builds one explainer per metric and plays every game,
    returning `{game: (importance, warning, interactions, moebius, total)}` by
    metric. Gains are measured from *baseline*, the space's default when
    omitted. A cancelled run, a run whose importance figures are both off
    (`BaseOptimizer.analytics_wanted`), and a run over budget skip the games,
    checked in the listed order.
    """
    if cancel_event is not None and cancel_event.is_set():
        return _skipped_games(
            metrics,
            "Importance analytics were skipped because the run was "
            "cancelled. Resuming the run computes them over every trial.")

    if not wanted:
        return _skipped_games(
            metrics,
            "Importance analytics were skipped: both figures that show "
            "them are switched off for this experiment. Switch one on and "
            "run again to compute them.")

    too_wide = eager_analytics_budget_exceeded(config_space, metrics, budget)
    if too_wide:
        return _skipped_games(metrics, too_wide)

    by_game: Dict[str, _GameResults] = {game: ({}, {}, {}, {}, {}) for game in HP_GAMES}
    # A list of names gives no direction for a file-declared metric, so a caller
    # holding declared metrics passes the mapping.
    directions = metrics if isinstance(metrics, Mapping) else {}
    for metric_name in metrics:
        explainer = _build_explainer(config_space, trials, metric_name, seed,
                                          metric=directions.get(metric_name))
        for game in HP_GAMES:
            importance, warning, interactions, moebius, total = compute_hp_game(
                config_space, trials, metric_name, game, seed, explainer=explainer,
                baseline=baseline)
            by_game[game][0][metric_name] = importance
            by_game[game][1][metric_name] = warning
            by_game[game][2][metric_name] = interactions
            by_game[game][3][metric_name] = moebius
            by_game[game][4][metric_name] = total
    return by_game


def _build_explainer(config_space, trials: List[TrialResult], metric_name: str,
                     seed: int = 0, metric=None):
    """Building the `ExplanationTask` fits the surrogate, the one expensive step
    the three games share. `insufficient` marks fewer than two trials, which get
    uniform weights, and a failed build sets `hs` to None and tries the forest.
    An explicit `base_model` carries *seed* into the surrogate.
    """
    from sklearn.ensemble import RandomForestRegressor

    data = _pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
    if len(data) < 2:
        return _MetricExplainer(
            None,
            "Not enough trials for importance estimation; showing uniform weights.",
            True)
    try:
        task = ExplanationTask.from_data(
            config_space, data,
            base_model=RandomForestRegressor(n_estimators=100, random_state=seed))
        return _MetricExplainer(HyperSHAP(task), None, False)
    except Exception as e:
        return _MetricExplainer(None, f"HyperSHAP failed to build a surrogate ({e})", False)


def compute_hp_game(
    config_space,
    trials: List[TrialResult],
    metric_name: str,
    game: str,
    seed: int = 0,
    explainer=None,
    baseline: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], Optional[str], Dict[str, Dict[str, float]], list, float]:
    """`compute_hp_game` plays *game* once for its Möbius decomposition, which
    stays the same at any truncation order, and derives order-2 FSII values with
    `MoebiusConverter` instead of a second 2^n_hp pass. `compute_hp_game` keeps
    order 2 because FSII terms are fitted jointly, and a higher order would
    change every stored importance. *total* is the raw order-1 sum, so `share x
    total` recovers units; the fallback rungs (forest importances, then uniform
    weights) return 0.0, no interactions and no Möbius terms.
    """
    names = list(config_space.keys())

    if explainer is None:
        explainer = _build_explainer(config_space, trials, metric_name, seed)
    build_warning = explainer.warning

    if explainer.insufficient:
        uniform = 1.0 / len(names) if names else 0.0
        return {p: uniform for p in names}, build_warning, {}, [], 0.0

    warning = None
    if explainer.hs is not None:
        hs = explainer.hs
        try:
            moebius_iv = getattr(hs, game)(
                baseline_config=_baseline_config(config_space, baseline),
                index="Moebius", order=len(names), seed=seed)
            fsii = hs.get_interaction_values_with_names(
                MoebiusConverter(moebius_iv)(index="FSII", order=2))
            moebius = _moebius_terms(hs.get_interaction_values_with_names(moebius_iv))
            raw = {p: abs(fsii.get((p,), 0.0)) for p in names}
            total = sum(raw.values())
            if not total:
                # All-zero contributions mean a saturated objective, which
                # differs from the uniform-weights rung. `compute_hp_game`
                # returns the importance empty, and every consumer shows its
                # fallback.
                return {}, (
                    f"HyperSHAP ({game}) found no signal in this run's trials: "
                    f"every hyperparameter's contribution was zero. This "
                    f"usually means the objective is saturated — try a "
                    f"harder dataset or a wider search space."), {}, [], 0.0
            importance = {k: v / total for k, v in raw.items()}
            interactions = _extract_pairwise(fsii, names)
            return importance, None, interactions, moebius, total
        except Exception as e:
            warning = f"HyperSHAP ({game}) failed ({e}) — falling back to surrogate feature importances."
    else:
        # `build_warning` already names HyperSHAP.
        warning = f"{build_warning} ({game}) — falling back to surrogate feature importances."

    rf, fit_warning = explainer.fallback_surrogate(
        config_space, trials, metric_name, seed)
    if rf is not None:
        importances = rf.feature_importances_
        total = importances.sum() or 1.0
        return ({p: float(importances[i] / total) for i, p in enumerate(names)},
                warning, {}, [], 0.0)
    warning += f" Surrogate fallback also failed ({fit_warning}); showing uniform weights."
    uniform = 1.0 / len(names) if names else 0.0
    return {p: uniform for p in names}, warning, {}, [], 0.0


def _extract_pairwise(named: Dict[tuple, float], names: List[str]) -> Dict[str, Dict[str, float]]:
    """`_extract_pairwise` builds a symmetric grid with signed order-1 values on
    the diagonal, raw so the heatmap keeps sign and scale.
    """
    grid = {p: {p: float(named.get((p,), 0.0))} for p in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            grid[a][b] = grid[b][a] = float(named.get((a, b), 0.0))
    return grid


#: Below `SETTLED` of the achievable gain, the remainder is rounding.
SETTLED = 0.02


def tuning_progress(result, metric, ablation):
    """Tunability decomposes best achievable minus default and the ablation
    decomposes the ablated trial minus default, both as order-1 FSII terms from
    the same zeroed baseline, so subtracting is valid. Achievable is a lower
    bound from 10,000 random draws, so `beyond` marks a search that beat the
    estimate, and order-1 terms miss the order-2 share. `tuning_progress`
    returns nothing for a result without `hyperparameter_tunability_total`. The
    result is `(rows, settled)`, with *settled* True when the remainder is
    rounding.
    """
    shares = result.hyperparameter_importance.get(metric, {})
    total = result.hyperparameter_tunability_total.get(metric, 0.0)
    if not shares or not total or not ablation:
        return [], False

    rows = []
    for name, share in shares.items():
        achievable = share * total
        # A negative value means the trial drifted worse than the default, the
        # one finding actionable alone.
        banked = ablation.get(name, 0.0)
        rows.append({
            "name": name,
            "achievable": achievable,
            "banked": banked,
            # A negative remainder is no remainder.
            "remaining": max(0.0, achievable - banked),
            "banked_share": (banked / achievable) if achievable else 0.0,
            # The search beat 10,000 random draws on this axis.
            "beyond": banked > achievable > 0,
        })
    rows.sort(key=lambda row: row["remaining"], reverse=True)
    left = sum(row["remaining"] for row in rows)
    return rows, left <= SETTLED * (total or 1.0)
