"""The figures an experiment page can show, in page order.

Each is a `Figure` subclass declaring how it is named, how wide it sits, and how
it draws — see `base.py` for the full set of options and how to add one.
"""

from django.utils.translation import gettext_lazy as _

from .base import FULL, HALF, Figure
from .plots import (
    hyperparameter_importance_plot,
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
    # default look.
    views = tuple(f"{game}-{rendering}" for game in HP_GAME_FIELDS for rendering in HP_RENDERINGS)

    @classmethod
    def plot(cls, result, metric=None, view=None):
        game, rendering = (view or cls.views[0]).split("-")
        importance_field, _ = HP_GAME_FIELDS[game]
        importance = getattr(result, importance_field).get(metric, {})
        return hyperparameter_importance_plot(importance, rendering)


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
    PerformanceOverTime,
    TrialDuration,
    Trials,
)

FIGURES_BY_KEY = {figure.key: figure for figure in FIGURES}
