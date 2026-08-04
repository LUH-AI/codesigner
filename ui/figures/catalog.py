"""The figures an experiment page can show, in page order.

Each is a `Figure` subclass declaring how it is named, how wide it sits, and how
it draws — see `base.py` for the full set of options and how to add one.
"""

from django.utils.translation import gettext_lazy as _

from .base import FULL, HALF, Figure
from .plots import (
    error_over_time_plot,
    hyperparameter_importance_plot,
    incumbent_performance_plot,
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


class HyperparameterImportance(Figure):
    # The heading keeps its existing wording: it is already translated, and
    # rewording it would orphan the de/es entries for no visible gain.
    key = "hyperparameter_importance"
    label = _("Hyperparameter importance (HyperSHAP)")
    per_metric = True

    @classmethod
    def plot(cls, result, metric=None):
        return hyperparameter_importance_plot(result, metric)


class IncumbentPerformance(Figure):
    key = "incumbent_performance"
    label = _("Performance of Incumbent")
    per_metric = True
    # Scores are bounded, so "absolute" is the full 0-1 range.
    absolute_scale = {"yaxis.range": [0, 1], "yaxis.autorange": False}

    @classmethod
    def plot(cls, result, metric=None):
        """Highlights the metric's best trial until another is clicked."""
        best_idx = max(range(len(result.trials)),
                       key=lambda i: result.trials[i].scores[metric])
        return incumbent_performance_plot(result, metric, selected_idx=best_idx)


class TrialDuration(Figure):
    """One bar per trial. The same for every metric, so it is drawn once."""

    key = "trial_duration"
    label = _("Trial duration")

    @classmethod
    def plot(cls, result, metric=None):
        return trial_duration_plot(result)


class ErrorOverTime(Figure):
    key = "error_over_time"
    label = _("Error over time")
    per_metric = True
    # The y-axis is log, so the full error range 1e-3..1 is log10 -3..0.
    absolute_scale = {"yaxis.range": [-3, 0], "yaxis.autorange": False}

    @classmethod
    def plot(cls, result, metric=None):
        return error_over_time_plot(result, metric)


class Trials(Figure):
    """Every trial as a sortable table — a column per metric needs the width."""

    key = "trials"
    label = _("Trials")
    width = FULL


FIGURES = (
    BestConfiguration,
    SelectedConfiguration,
    HyperparameterImportance,
    IncumbentPerformance,
    TrialDuration,
    ErrorOverTime,
    Trials,
)

FIGURES_BY_KEY = {figure.key: figure for figure in FIGURES}
