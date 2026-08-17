"""The figures an experiment page can show, in page order.

Each is a `Figure` subclass declaring how it is named, how wide it sits, and how
it draws — see `base.py` for the full set of options and how to add one.
"""

from django.utils.translation import gettext_lazy as _

from .base import FULL, HALF, Figure
from .plots import (
    configuration_cube_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    performance_over_time_plot,
    trial_duration_plot,
)


class BestConfiguration(Figure):
    """The best trial for the viewed metric, as a table. No plot."""

    key = "best_configuration"
    label = _("Best configuration")


class SelectedConfiguration(Figure):
    """The trial clicked on the performance figure, as a table. No plot."""

    key = "selected_configuration"
    label = _("Selected configuration")


#: Which OptimizationResult field pair backs each HyperSHAP "explanation
#: game" this app surfaces on the importance figure — see
#: core.optimizers.base.BaseOptimizer.HP_GAMES for what each one answers.
#: Order here is the games' order in the game selector.
HP_GAME_FIELDS = {
    "tunability": ("hyperparameter_importance", "hyperparameter_importance_warning"),
    "sensitivity": ("hyperparameter_sensitivity", "hyperparameter_sensitivity_warning"),
    "mistunability": ("hyperparameter_mistunability", "hyperparameter_mistunability_warning"),
}

#: The three ways to draw one game's numbers — see hyperparameter_importance_plot.
HP_RENDERINGS = ("pie", "bar", "table")


class HyperparameterImportance(Figure):
    # The heading keeps its existing wording: it is already translated, and
    # rewording it would orphan the de/es entries for no visible gain.
    key = "hyperparameter_importance"
    label = _("Hyperparameter importance (HyperSHAP)")
    per_metric = True
    # Two independent choices compose into one flat view key (see
    # experiment_detail.html's game/rendering selects): which game, and how to
    # draw it. Tunability + pie is first, matching the figure's pre-existing
    # default look. "local-bar" is a fourth game, ablation against the
    # currently-selected trial, tacked on rather than a fourth row of the
    # game x rendering product — it only has one rendering (see
    # hyperparameter_ablation_plot) and, unlike the other three, its value
    # depends on which trial is selected, not just the metric, so `plot()`
    # below can't compute it — that needs the live model/config space, which
    # only ui/views.py's _detail_context (default: the metric's best trial)
    # and the trial_ablation endpoint (a click) have access to. This entry
    # exists so `views`/`figure_views` still list it as a real option.
    views = tuple(f"{game}-{rendering}" for game in HP_GAME_FIELDS for rendering in HP_RENDERINGS) \
        + ("local-bar",)

    @classmethod
    def plot(cls, result, metric=None, view=None):
        view = view or cls.views[0]
        game, rendering = view.split("-")
        if game == "local":
            return None
        importance_field, _warning_field = HP_GAME_FIELDS[game]
        importance = getattr(result, importance_field).get(metric, {})
        return hyperparameter_importance_plot(importance, rendering)


class HyperparameterInteractions(Figure):
    """Pairwise (order-2) HyperSHAP tunability interactions — a heatmap by
    default, a bar of the top-10 strongest pairs as the alternate view.

    Free byproduct of the importance figure's own HyperSHAP call: tunability
    is computed with order 2 by default already, so this reuses
    `OptimizationResult.hyperparameter_interactions` rather than asking
    HyperSHAP for anything new. Sensitivity/mistunability's own interaction
    grids are computed the same way (see BaseOptimizer.compute_hp_games) but
    have no field or view yet — only tunability's is wired up here.
    """

    key = "hyperparameter_interactions"
    label = _("Hyperparameter interactions (HyperSHAP)")
    per_metric = True
    views = ("heatmap", "bar")

    @classmethod
    def plot(cls, result, metric=None, view=None):
        view = view or cls.views[0]
        interactions = result.hyperparameter_interactions.get(metric, {})
        if view == "bar":
            return hyperparameter_interactions_bar_plot(interactions)
        return hyperparameter_interactions_heatmap_plot(interactions)


class PerformanceOverTime(Figure):
    """Every trial's outcome and the running-best line — what used to be two
    figures (performance-over-trials, error-over-time) are four views of one
    curve here: trial index or elapsed time on x, score or error on y."""

    key = "performance_over_time"
    label = _("Performance over time")
    per_metric = True
    views = ("trial-score", "trial-error", "time-score", "time-error")
    # Keyed by view, since the sensible "absolute" range depends on which
    # y-axis is showing: scores are bounded 0-1 (linear); error is log, so its
    # full range 1e-3..1 is log10 -3..0. Same pair for both x-axis choices.
    absolute_scale = {
        "trial-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "trial-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
        "time-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "time-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
    }

    @classmethod
    def plot(cls, result, metric=None, view=None):
        """Highlights the metric's best trial until another is clicked."""
        view = view or cls.views[0]
        x_axis, y_axis = view.split("-")
        best_idx = (max(range(len(result.trials)), key=lambda i: result.trials[i].scores[metric])
                    if result.trials else None)
        return performance_over_time_plot(
            result, metric, x_axis=x_axis, y_axis=y_axis, selected_idx=best_idx)


class ConfigurationCube(Figure):
    """Every trial as one point in hyperparameter space, colored by score —
    DeepCave's "Configuration Cube," with two or three actual hyperparameters
    as the axes rather than an MDS projection (see configuration_cube_plot).

    No `views`: which hyperparameters are on which axis is a per-experiment,
    unbounded combination, not a fixed enumerable set the server can
    precompute one JSON entry per option for. Instead `plot()` ships one
    default 2D scatter, and every hyperparameter's values ride along in the
    trace's own `customdata` for the client to remap without a server round
    trip — see experiment_detail.html's applyCubeAxes.
    """

    key = "configuration_cube"
    label = _("Configuration cube")
    width = FULL
    per_metric = True

    @classmethod
    def plot(cls, result, metric=None):
        return configuration_cube_plot(result, metric)


class TrialDuration(Figure):
    """One bar per trial. The same for every metric, so it is drawn once."""

    key = "trial_duration"
    label = _("Trial duration")

    @classmethod
    def plot(cls, result, metric=None):
        return trial_duration_plot(result)


class Trials(Figure):
    """Every trial as a sortable table — a column per metric needs the width."""

    key = "trials"
    label = _("Trials")
    width = FULL


FIGURES = (
    BestConfiguration,
    SelectedConfiguration,
    HyperparameterImportance,
    HyperparameterInteractions,
    PerformanceOverTime,
    ConfigurationCube,
    TrialDuration,
    Trials,
)

FIGURES_BY_KEY = {figure.key: figure for figure in FIGURES}
