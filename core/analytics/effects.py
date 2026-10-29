from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hypershap import ExplanationTask, HyperSHAP

from ..metrics import metric_for
from .games import baseline_config, build_explainer
from .surrogate import (
    fit_surrogate, hp_grid, pair_trials_with_oriented_scores, predict_with_spread,
    sample_evenly,
)

if TYPE_CHECKING:
    from ..optimizers.base import TrialResult


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

    data = pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
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
            baseline_config=baseline_config(config_space, baseline))
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
    caps the cost, sampled by `sample_evenly`.
    """
    explainer = build_explainer(config_space, trials, metric_name, seed,
                                      metric=metric)
    if explainer.hs is None:
        return [], [], (explainer.warning
                        or "Not enough trials for local explanations.")

    names = list(config_space.keys())
    rows = []
    # The sample skips trials, and the page selects by position in `trials` (see
    # `_selection_meta`).
    for index, trial in sample_evenly(list(enumerate(trials)), max_trials):
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

    grid = hp_grid(config_space[hp_name], n_points)

    trials = sample_evenly(trials, max_ice_curves)

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
    x_grid = hp_grid(config_space[x_hp], n_points)
    y_grid = hp_grid(config_space[y_hp], n_points)
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

    _, spread = predict_with_spread(rf, np.array(rows))
    for (j, i), value in zip(slots, spread):
        z[j][i] = float(value)

    return x_grid, y_grid, z, None
