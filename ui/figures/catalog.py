"""Each figure here is a `Figure` subclass, and `base.py` lists the options and
how to add one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _

from core.optimizers.base import HP_GAME_FIELDS
from core.projection import METHODS

from .base import ACQUISITION, HYPERSHAP, LONG, MISC, TABLE, Figure
from .plots import (
    configuration_cube_plot,
    configuration_projection_plot,
    hyperparameter_importance_plot,
    hyperparameter_interactions_bar_plot,
    hyperparameter_interactions_heatmap_plot,
    hyperparameter_graph_plot,
    hyperparameter_orders_plot,
    hyperparameter_upset_plot,
    parallel_coordinates_plot,
    performance_over_time_plot,
    trial_duration_plot,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go


class BestConfiguration(Figure):
    """`BestConfiguration` tabulates the viewed metric's best trial."""

    key = "best_configuration"
    label = _("Best configuration")
    home_tab = MISC


class SelectedConfiguration(Figure):
    """The sidebar holds the selected trial's table, so it stays readable while
    the reader clicks figures.
    """

    key = "selected_configuration"
    label = _("Selected Configuration")
    in_sidebar = True


#: The selector shows the game names and questions, without the MAX/VAR/MIN
#: aggregation names.
HP_GAME_LABELS = {
    "tunability": _("Tunability"),
    "mistunability": _("Mistunability"),
    "sensitivity": _("Sensitivity"),
}
HP_GAME_HELP = {
    "tunability": _("How much is there to gain by tuning this hyperparameter?"),
    "mistunability": _("How much is there to lose by getting this hyperparameter wrong?"),
    "sensitivity": _("How much does performance move as this hyperparameter moves?"),
}


def game_field(result, game: str, part: str, metric, default):
    """`game_field` makes "which game" one lookup for every HyperSHAP figure.
    """
    return getattr(result, HP_GAME_FIELDS[game][part]).get(metric, default)

#: Ways to draw a game; see hyperparameter_importance_plot.
HP_RENDERINGS = ("pie", "bar", "table")


class HyperparameterImportance(Figure):
    # Rewording the heading would orphan its de/es translations.
    key = "hyperparameter_importance"
    label = _("Hyperparameter importance (HyperSHAP)")
    home_tab = HYPERSHAP
    per_metric = True
    explains_games = True
    # The page computes games against any non-default baseline on request, and
    # the interaction figures share the declaration.
    deferred = (("hypershap_baseline", _("HyperSHAP against the incumbent or selected trial")),)
    # The view key composes the page-wide game with the importance figure's own
    # rendering, and tunability plus pie comes first.
    views = tuple(f"{game}-{rendering}"
                  for game in HP_GAME_FIELDS for rendering in HP_RENDERINGS)

    @classmethod
    def plot(cls, result, metric=None, view=None):
        view = view or cls.views[0]
        game, rendering = view.split("-")
        return hyperparameter_importance_plot(
            game_field(result, game, "importance", metric, {}), rendering)


class LocalExplanation(Figure):
    """`LocalExplanation` explains one trial, so its value depends on the
    selection and needs the live model. The page fetches it, and `plot()` stays
    None.
    """

    key = "local_explanation"
    label = _("Local explanation (selected trial)")
    home_tab = HYPERSHAP
    per_metric = True
    # The stored settings key keeps the computation's name.
    deferred = (("local_ablation", _("Local explanation (selected trial)")),)


class _Interactions(Figure):
    """The five interaction figures sit side by side, each answering a different
    question. The heatmap and top pairs read the order-2 FSII grid, and the
    others read the Möbius decomposition (see `core.analytics.compute_hp_game`).
    All of them come free with the importance figure's HyperSHAP call, and the
    sidebar's game selector switches their `views` together.
    """

    home_tab = HYPERSHAP

    per_metric = True
    explains_games = True
    views = tuple(HP_GAME_FIELDS)
    #: The game field read: "interactions" (order-2 FSII) or "moebius".
    source = "interactions"
    #: `staticmethod` keeps the builder a plain function.
    builder: Callable[..., go.Figure | None] | None = None

    @classmethod
    def plot(cls, result, metric=None, view=None):
        game = view or cls.views[0]
        empty = [] if cls.source == "moebius" else {}
        assert cls.builder is not None
        return cls.builder(game_field(result, game, cls.source, metric, empty))


class InteractionsHeatmap(_Interactions):
    """`InteractionsHeatmap` shows all pairs."""

    key = "interactions_heatmap"
    label = _("Interactions: heatmap")
    builder = staticmethod(hyperparameter_interactions_heatmap_plot)


class InteractionsTopPairs(_Interactions):
    """The top pairs, ranked."""

    key = "interactions_top_pairs"
    label = _("Interactions: top pairs")
    builder = staticmethod(hyperparameter_interactions_bar_plot)


class InteractionsGraph(_Interactions):
    """`InteractionsGraph` gives the only reading above order 2."""

    key = "interactions_graph"
    label = _("Interactions: graph")
    source = "moebius"
    builder = staticmethod(hyperparameter_graph_plot)


class InteractionsCoalitions(_Interactions):
    """`InteractionsCoalitions` draws an UpSet plot, legible past the graph's
    six-hyperparameter limit."""

    key = "interactions_coalitions"
    label = _("Interactions: coalitions")
    source = "moebius"
    builder = staticmethod(hyperparameter_upset_plot)


class InteractionsOrders(_Interactions):
    """Where each hyperparameter's value comes from, by order."""

    key = "interactions_orders"
    label = _("Interactions: by order")
    source = "moebius"
    builder = staticmethod(hyperparameter_orders_plot)


class PerformanceOverTime(Figure):
    """Four views plot each trial's outcome and the running best by trial or
    time."""

    key = "performance_over_time"
    home_tab = MISC
    shape = LONG
    # See `Figure.live`.
    live = True
    # The label names the trial as the unit, and the key stays for stored
    # settings.
    label = _("Trial performance")
    # Clicks here select a trial, as in other figures.
    selects_trials = True
    per_metric = True
    views = ("trial-score", "trial-error", "time-score", "time-error")
    # `absolute_scale_for` supplies the real ranges; this enables the toggle.
    absolute_scale = {
        "trial-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "trial-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
        "time-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "time-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
    }

    @classmethod
    def absolute_scale_for(cls, metric):
        """Each view's absolute range comes from the metric's bounds, and a view
        without one is dropped. The error axis is log10 of
        `core.metrics.error_range`, floored at 1e-3 like the figure. An empty
        dict means no toggle.
        """
        import math

        from core.metrics import error_range

        scales = {}
        if metric.low is not None and metric.high is not None:
            scales["score"] = {"yaxis.range": [metric.low, metric.high],
                               "yaxis.autorange": False}
        span = error_range(metric)
        if span is not None:
            lo, hi = span
            if hi > 0:
                scales["error"] = {
                    "yaxis.range": [math.log10(max(1e-3, lo)), math.log10(hi)],
                    "yaxis.autorange": False}
        return {view: scales[view.split("-")[1]]
                for view in cls.views if view.split("-")[1] in scales}

    @classmethod
    def plot(cls, result, metric=None, view=None):
        """`plot` initially highlights the best trial."""
        view = view or cls.views[0]
        x_axis, y_axis = view.split("-")
        return performance_over_time_plot(
            result, metric, x_axis=x_axis, y_axis=y_axis,
            selected_idx=result.best_index(metric))


class ConfigurationCube(Figure):
    """The axes view uses chosen hyperparameters as axes, while PCA and PLS
    project the space so distances mean something. All three views come with the
    page, since a projection costs 0.2 to 5 ms. The client builds axes from
    `customdata` without a round trip.
    """

    key = "configuration_cube"
    home_tab = ACQUISITION
    # See `Figure.live`.
    live = True
    # "Projection" covers all three views, and the key stays for stored
    # settings.
    label = _("Hyperparameter space projection")
    per_metric = True
    selects_trials = True
    # The page opens on PLS, since PLS shows which directions moved the metric.
    views = ("axes",) + METHODS
    default_view = "pls"
    # A logarithmic hyperparameter needs a log axis, and the projections need
    # the space as well.
    needs_config_space = True
    #: Surrogate uncertainty is per hyperparameter pair, too many to precompute,
    #: and covers the axes view only.
    deferred = (("surrogate_uncertainty", _("Surrogate uncertainty (axes view)")),)

    @classmethod
    def plot(cls, result, metric=None, view=None, config_space=None):
        view = view or cls.opening_view()
        if view in METHODS:
            return configuration_projection_plot(
                result, metric, view, config_space=config_space)
        return configuration_cube_plot(result, metric, config_space=config_space)


class ParallelCoordinates(Figure):
    """Each trial is a line across its hyperparameters ending at its score,
    ordered by HyperSHAP tunability (see `parallel_coordinates_plot`).
    """

    key = "parallel_coordinates"
    home_tab = MISC
    shape = LONG
    # See `Figure.live`.
    live = True
    label = _("Parallel coordinates")
    selects_trials = True
    per_metric = True
    # Parcoords lacks log axes, so log values are recoded and relabelled.
    needs_config_space = True

    @classmethod
    def plot(cls, result, metric=None, view=None, config_space=None):
        return parallel_coordinates_plot(result, metric, config_space=config_space)


class PartialDependence(Figure):
    """The page fetches one hyperparameter's PDP and ICE curves on demand from
    the `partial_dependence` endpoint. Precomputing every hyperparameter would
    waste each page load, so `plot()` stays None.
    """

    key = "partial_dependence"
    label = _("Partial dependence (PDP/ICE)")
    home_tab = MISC
    shape = LONG
    per_metric = True
    # The whole figure is deferred, so the computation shares the key.
    deferred = (("partial_dependence", _("Partial dependence (PDP/ICE)")),)


class AcquisitionSlice(Figure):
    """The acquisition curve shows a prior's effect, since the prior multiplies
    the acquisition. The slice is fetched per metric and hyperparameter, and
    `ui/static/ui/acquisition.js` draws it with `Prior` from one request.
    """

    key = "acquisition_slice"
    label = _("Acquisition function")
    home_tab = ACQUISITION
    shape = LONG
    per_metric = True
    # `Prior` reads it too; the stored name stays.
    deferred = (("acquisition_slice", _("Acquisition and priors")),)


class Prior(Figure):
    """A stated or freehand belief about one hyperparameter weights the
    acquisition, with a decay over the run. `AcquisitionSlice`'s script draws
    it, and it applies across metrics and before any run.
    """

    key = "prior"
    label = _("Prior")
    home_tab = ACQUISITION
    shape = LONG


class LocalEffects(Figure):
    """A beeswarm of sampled trials' local ablations needs one game per trial.
    The page fetches it on request, capped by `local_effects_max_trials`.
    """

    key = "local_effects"
    label = _("Local effects across trials")
    # Local explanation's game, across the run.
    home_tab = HYPERSHAP
    shape = LONG
    # Most sampled trials have no point here, so the plot names its trials (see
    # `_selection_meta`).
    selects_trials = True
    per_metric = True
    deferred = (("local_effects", _("Local effects across trials")),)


class TrialDuration(Figure):
    """`TrialDuration` draws one figure for every metric."""

    key = "trial_duration"
    home_tab = MISC
    # See `Figure.live`.
    live = True
    label = _("Trial duration")
    selects_trials = True

    @classmethod
    def plot(cls, result, metric=None, view=None):
        return trial_duration_plot(result)


class Trials(Figure):
    """`Trials` shows a sortable table with one column per metric."""

    key = "trials"
    label = _("Trials")
    home_tab = MISC
    shape = TABLE
    #: `page_size` sets the rows per page for the template field and its script,
    #: and readers can change it on the page.
    page_size = 50
    # Rows carry the trial index, and the highlight is a class.
    selects_trials = True


class Forecast(Figure):
    """The page fetches the forecast per trial, since drawing it reruns the
    model once per backtest.
    """

    key = "forecast"
    label = _("Forecast")
    home_tab = MISC
    shape = LONG
    forecast_only = True
    deferred = (("forecast", _("Forecast")),)


FIGURES = (
    SelectedConfiguration,
    BestConfiguration,
    HyperparameterImportance,
    PerformanceOverTime,
    InteractionsHeatmap,
    InteractionsTopPairs,
    InteractionsGraph,
    InteractionsCoalitions,
    InteractionsOrders,
    TrialDuration,
    Forecast,
    ParallelCoordinates,
    PartialDependence,
    LocalExplanation,
    LocalEffects,
    AcquisitionSlice,
    Prior,
    ConfigurationCube,
    Trials,
)

FIGURES_BY_KEY = {figure.key: figure for figure in FIGURES}
