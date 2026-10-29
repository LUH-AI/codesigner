from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hypershap import ExplanationTask, HyperSHAP
from shapiq.game_theory import MoebiusConverter

from .surrogate import fit_surrogate, pair_trials_with_oriented_scores

if TYPE_CHECKING:
    from ..optimizers.base import TrialResult


#: Tunability (MAX) is the upside of tuning a hyperparameter, sensitivity (VAR)
#: is how much performance varies with it, and mistunability (MIN) is the
#: downside of a bad value. `optimizer_bias` needs an ensemble of optimizers, so
#: the app omits it.
HP_GAMES = ("tunability", "sensitivity", "mistunability")

#: 1024 coalitions admit every built-in model, the six-hyperparameter SVM
#: included (960 coalitions, 1,344 with probability metrics), and turn away a
#: 10-hyperparameter upload (12,288 coalitions, about 3 minutes). Deployments
#: override the default through `ANALYTICS_EAGER_MAX_COALITIONS`, and None means
#: no limit.
EAGER_MAX_COALITIONS = 1024


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


def baseline_config(config_space, baseline: Optional[Dict[str, Any]]):
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
        explainer = build_explainer(config_space, trials, metric_name, seed,
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


def build_explainer(config_space, trials: List[TrialResult], metric_name: str,
                     seed: int = 0, metric=None):
    """Building the `ExplanationTask` fits the surrogate, the one expensive step
    the three games share. `insufficient` marks fewer than two trials, which get
    uniform weights, and a failed build sets `hs` to None and tries the forest.
    An explicit `base_model` carries *seed* into the surrogate.
    """
    from sklearn.ensemble import RandomForestRegressor

    data = pair_trials_with_oriented_scores(config_space, trials, metric_name, metric)
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
        explainer = build_explainer(config_space, trials, metric_name, seed)
    build_warning = explainer.warning

    if explainer.insufficient:
        uniform = 1.0 / len(names) if names else 0.0
        return {p: uniform for p in names}, build_warning, {}, [], 0.0

    warning = None
    if explainer.hs is not None:
        hs = explainer.hs
        try:
            moebius_iv = getattr(hs, game)(
                baseline_config=baseline_config(config_space, baseline),
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
