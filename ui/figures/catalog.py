"""The figures an experiment page can show, in page order.

Each is a `Figure` subclass declaring how it is named, how wide it sits, and how
it draws — see `base.py` for the full set of options and how to add one.
"""

from django.utils.translation import gettext_lazy as _

from core.projection import METHODS

from .base import ACQUISITION, HYPERSHAP, LONG, MISC, SQUAT, TABLE, Figure
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


class BestConfiguration(Figure):
    """The best trial for the viewed metric, as a table. No plot."""

    key = "best_configuration"
    label = _("Best configuration")
    home_tab = MISC
    pinned = True


class SelectedConfiguration(Figure):
    """The selected trial, as a table. No plot.

    Lives in the sidebar: it is what every other figure's click is asking about,
    so it has to be readable while you are clicking rather than wherever the
    page happens to be scrolled to.
    """

    key = "selected_configuration"
    label = _("Selected Configuration")
    in_sidebar = True


#: Which OptimizationResult fields back each HyperSHAP "explanation game" —
#: see core.optimizers.base.BaseOptimizer.HP_GAMES for what each one answers.
#: Order here is the games' order in the page's one game selector.
#:
#: Four fields each, because a game is not only an importance: the same call
#: produces the order-2 interaction grid and the full Möbius decomposition, and
#: every figure that reads any of them reads the game the sidebar names. The
#: tunability entry's names have no game in them — they were the only ones
#: stored back when only tunability's had anywhere to go, and they are in every
#: .ihpo ever written.
HP_GAME_FIELDS = {
    "tunability": {
        "importance": "hyperparameter_importance",
        "warning": "hyperparameter_importance_warning",
        "interactions": "hyperparameter_interactions",
        "moebius": "hyperparameter_moebius",
    },
    # Mistunability before sensitivity: it is tunability's mirror — what there
    # is to lose against what there is to gain — and the two read as a pair.
    "mistunability": {
        "importance": "hyperparameter_mistunability",
        "warning": "hyperparameter_mistunability_warning",
        "interactions": "hyperparameter_mistunability_interactions",
        "moebius": "hyperparameter_mistunability_moebius",
    },
    "sensitivity": {
        "importance": "hyperparameter_sensitivity",
        "warning": "hyperparameter_sensitivity_warning",
        "interactions": "hyperparameter_sensitivity_interactions",
        "moebius": "hyperparameter_sensitivity_moebius",
    },
}

#: What each game is called where a reader sees it, and what it asks. The
#: selector's own labels; the keys above are what the code and the stored result
#: use. The questions are BaseOptimizer.HP_GAMES' own, said without the
#: aggregation names — MAX/VAR/MIN mean nothing to someone reading a figure.
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
    """One game's *part* for *metric*, or *default*.

    The indirection every HyperSHAP figure goes through, so that "which game"
    is one lookup rather than a branch in each of them.
    """
    return getattr(result, HP_GAME_FIELDS[game][part]).get(metric, default)

#: The three ways to draw one game's numbers — see hyperparameter_importance_plot.
HP_RENDERINGS = ("pie", "bar", "table")


class HyperparameterImportance(Figure):
    # The heading keeps its existing wording: it is already translated, and
    # rewording it would orphan the de/es entries for no visible gain.
    key = "hyperparameter_importance"
    label = _("Hyperparameter importance (HyperSHAP)")
    home_tab = HYPERSHAP
    pinned = True
    per_metric = True
    # Two independent choices compose into one flat view key: which game — now
    # chosen once for the whole page, in the sidebar — and how to draw it, which
    # is this figure's own and stays on it. Tunability + pie is first, matching
    # the figure's pre-existing default look.
    views = tuple(f"{game}-{rendering}"
                  for game in HP_GAME_FIELDS for rendering in HP_RENDERINGS)

    @classmethod
    def plot(cls, result, metric=None, view=None):
        view = view or cls.views[0]
        game, rendering = view.split("-")
        return hyperparameter_importance_plot(
            game_field(result, game, "importance", metric, {}), rendering)


class LocalExplanation(Figure):
    """The selected trial's score, built from the config-space default one
    hyperparameter at a time — HyperSHAP's ablation game.

    Its own figure rather than a fourth option on the game selector, because it
    is a different question about a different subject: the other three explain
    the search, this explains one trial, and it has no interactions to give the
    five interaction figures. It also cannot be precomputed the way they are —
    its value depends on which trial is selected, not just the metric, and it
    needs the live model and config space, which only `ui/views.py`'s
    `_detail_context` (defaulting to the metric's best trial) and the
    `trial_ablation` endpoint (a click) have. So `plot()` stays None and the
    page fetches it, like partial dependence and local effects.
    """

    key = "local_explanation"
    label = _("Local explanation (selected trial)")
    home_tab = HYPERSHAP
    default_columns = 2
    per_metric = True
    # The computation's name, not the figure's, and unchanged: it is stored in
    # settings, and it was already named for the computation back when this was
    # one view of the importance figure.
    deferred = (("local_ablation", _("Local explanation (selected trial)")),)


class _Interactions(Figure):
    """One way of reading the same set of hyperparameter interactions.

    The five below are five figures rather than five views of one, because they
    answer different questions and are wanted side by side: the heatmap is a
    grid you scan, the graph is a shape you recognise, the coalitions are a
    ranked list you read. Behind a single selector only one could ever be on the
    page at a time, and each now has its own visibility setting.

    They are also two different computations wearing one name. The heatmap and
    the top pairs read the order-2 FSII grid; the graph, the coalitions and the
    orders read the Möbius decomposition, which carries every coalition of every
    size. Those are not the same numbers and deliberately so — see
    `BaseOptimizer._shared_exact_computer`.

    All of it is a free byproduct of the importance figure's own HyperSHAP call
    (order 2 is already its default), so nothing here is fetched on demand —
    and that is true of all three games, which is what lets one selector drive
    every figure here as well as the importance figure. The games are these
    figures' `views`: a game is a different set of numbers to draw rather than a
    different way to draw them, but the page's view machinery is exactly the
    "one precomputed payload per option" plumbing that needs, so they use it.
    Nothing switches them individually — the sidebar's selector switches all of
    them at once (see experiment_detail.html).
    """

    home_tab = HYPERSHAP

    per_metric = True
    views = tuple(HP_GAME_FIELDS)
    #: Which of a game's fields this reading comes out of: "interactions" for
    #: the order-2 FSII grid, "moebius" for the full decomposition.
    source = "interactions"
    #: The builder. `staticmethod` so it stays a plain function rather than
    #: becoming a method of the figure that happens to take a dict.
    builder = None

    @classmethod
    def plot(cls, result, metric=None, view=None):
        game = view or cls.views[0]
        empty = [] if cls.source == "moebius" else {}
        return cls.builder(game_field(result, game, cls.source, metric, empty))


class InteractionsHeatmap(_Interactions):
    """Hyperparameter × hyperparameter, every pair at once."""

    key = "interactions_heatmap"
    label = _("Interactions: heatmap")
    builder = staticmethod(hyperparameter_interactions_heatmap_plot)


class InteractionsTopPairs(_Interactions):
    """The handful of pairs that matter, ranked, without scanning a grid."""

    key = "interactions_top_pairs"
    label = _("Interactions: top pairs")
    builder = staticmethod(hyperparameter_interactions_bar_plot)


class InteractionsGraph(_Interactions):
    """The Möbius graph — the only reading that shows anything above order 2."""

    key = "interactions_graph"
    label = _("Interactions: graph")
    source = "moebius"
    builder = staticmethod(hyperparameter_graph_plot)


class InteractionsCoalitions(_Interactions):
    """UpSet: which named combinations carry value. Stays legible where the
    graph stops, which is around six hyperparameters."""

    key = "interactions_coalitions"
    label = _("Interactions: coalitions")
    source = "moebius"
    builder = staticmethod(hyperparameter_upset_plot)


class InteractionsOrders(_Interactions):
    """Alone, in pairs, or in larger groups — where each hyperparameter's
    attributed value actually comes from."""

    key = "interactions_orders"
    label = _("Interactions: by order")
    source = "moebius"
    builder = staticmethod(hyperparameter_orders_plot)


class PerformanceOverTime(Figure):
    """Every trial's outcome and the running-best line — what used to be two
    figures (performance-over-trials, error-over-time) are four views of one
    curve here: trial index or elapsed time on x, score or error on y."""

    key = "performance_over_time"
    home_tab = MISC
    shape = LONG
    pinned = True
    # Drawn from the trials alone, so it can be updated mid-run. See Figure.live.
    live = True
    # Renamed from "Performance over time": every figure here is over time in
    # some sense, and what distinguishes this one is that its unit is the
    # trial. The key is unchanged — it is stored in settings, so renaming that
    # would be a data migration.
    label = _("Trial performance")
    # The figure the click-to-select gesture started on, and still the most
    # natural place to make it — but no longer the only one.
    selects_trials = True
    # Full width: the chart every other figure is read against, and the one
    # most often clicked. A spanning grid item always begins a fresh row, so it
    # lands below whatever tiles precede it however many are switched off —
    # which an ordering alone would not guarantee.
    per_metric = True
    views = ("trial-score", "trial-error", "time-score", "time-error")
    # Present so the toggle is offered at all; the ranges in it are the four
    # original metrics', and `absolute_scale_for` is what actually answers.
    absolute_scale = {
        "trial-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "trial-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
        "time-score": {"yaxis.range": [0, 1], "yaxis.autorange": False},
        "time-error": {"yaxis.range": [-3, 0], "yaxis.autorange": False},
    }

    @classmethod
    def absolute_scale_for(cls, metric):
        """The four views' absolute ranges under *metric*, or None for none.

        Keyed by view, since the sensible "absolute" range depends on which
        y-axis is showing — and now by metric too, since a range that is right
        for an accuracy is a fiction for an RMSE. Both axes resolve from the
        metric's own bounds, and either can come back empty:

        - **score** is linear over `[low, high]`, and needs both. An unbounded
          metric has no full range to pin to, so it offers no toggle.
        - **error** is `core.metrics.error_range`, drawn on a log axis, so the
          pinned range is its log10. `1e-3` is the floor the figure itself
          applies to keep a log axis off zero, and it is the same number here.

        A view with no range is dropped rather than set to None, so a metric
        with neither leaves an empty dict — which the page reads as "no toggle".
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
        """Highlights the metric's best trial until another is clicked."""
        view = view or cls.views[0]
        x_axis, y_axis = view.split("-")
        return performance_over_time_plot(
            result, metric, x_axis=x_axis, y_axis=y_axis,
            selected_idx=result.best_index(metric))


class ConfigurationCube(Figure):
    """Where the search went, three ways.

    **Axes** is DeepCave's "Configuration Cube": two or three actual
    hyperparameters as the axes, chosen by the reader, so a position on an axis
    is a literal hyperparameter value. That also means a *distance* between two
    points is not one — the axes carry unrelated units — which is what the other
    two views add.

    **PCA** and **PLS** project the whole configuration space down to two or
    three components, where a distance is a distance: PCA along the directions
    the trials varied in most, PLS along the directions that move the metric.
    See `core.projection`, which also says why a categorical is one-hot here and
    integer-coded on parallel coordinates.

    All three ship with the page. A projection costs 0.2–5 ms, from twenty
    trials to five thousand — a fetch and a Compute button would take longer to
    press than to compute.

    The reader's choices are not `views`, though the method is: which
    hyperparameters are on which axis, and how many components to draw, are
    per-experiment and unbounded, so every payload here carries its numbers in
    `customdata` and the client assembles the axes from them without a round
    trip. See `configuration_cube_plot` and experiment_detail.html's
    applyCubeAxes, which draws all three.
    """

    key = "configuration_cube"
    home_tab = ACQUISITION
    # Squat, but a plane rather than a tile: a scatter over a projected space
    # needs the room to be read as one.
    default_columns = 2
    pinned = True
    # Drawn from the trials alone, so it can be updated mid-run. See Figure.live.
    live = True
    # Renamed from "Configuration cube", which now names one of its three views.
    # "Projection" covers all three: each keeps two or three linear coordinates
    # of hyperparameter space and drops the rest, differing only in which
    # subspace they keep. The key is unchanged — it is stored in settings.
    label = _("Hyperparameter space projection")
    # Two rows as well as two columns. Every one of its three views is a scatter
    # over a space with no privileged direction, and a scatter squeezed into a
    # single row is a strip: the vertical axis gets a fifth of the room the
    # horizontal one does and reports a fifth of what it has to say.
    per_metric = True
    selects_trials = True
    # Listed simplest first, because that is how they read in the selector, and
    # opened on the last of them: PLS answers the question the page is actually
    # about — which directions through this space moved the metric — and axes is
    # the one that needs three decisions before it says anything at all.
    views = ("axes",) + METHODS
    default_view = "pls"
    # For the axes view: a hyperparameter the search treats logarithmically
    # needs a logarithmic axis, and the stored result carries values, not the
    # space they were drawn from. The projections need it too, for the same
    # reason one column further back — see `encode_configurations`.
    needs_config_space = True
    #: How unsure the surrogate is across the plane the axes view spans, drawn
    #: as a field under the trials. Deferred for the same reason partial
    #: dependence is: it is per *pair* of hyperparameters, which is not a small
    #: precomputable set — a model with six of them has fifteen pairs, and the
    #: reader looks at one. See `BaseOptimizer.compute_surrogate_uncertainty`,
    #: and `views.surrogate_uncertainty` for why it is the axes view only.
    deferred = (("surrogate_uncertainty", _("Surrogate uncertainty (axes view)")),)

    @classmethod
    def plot(cls, result, metric=None, view=None, config_space=None):
        view = view or cls.opening_view()
        if view in METHODS:
            return configuration_projection_plot(
                result, metric, view, config_space=config_space)
        return configuration_cube_plot(result, metric, config_space=config_space)


class ParallelCoordinates(Figure):
    """Every trial as one line across its hyperparameters, ending at its
    score — see parallel_coordinates_plot for why the axis order is
    HyperSHAP tunability rather than DeepCave's own fANOVA ordering.

    No `views`, no special client-side wiring: unlike the cube, axis order is
    a full, fixed ranking (not a per-experiment unbounded combination), so
    one precomputed plot per metric — the ordinary per_metric contract every
    figure before Phase 3 already used — is enough.
    """

    key = "parallel_coordinates"
    home_tab = MISC
    shape = LONG
    # Drawn from the trials alone, so it can be updated mid-run. See Figure.live.
    live = True
    label = _("Parallel coordinates")
    selects_trials = True
    per_metric = True
    # Same reason as the cube's, different remedy — Parcoords has no log axis,
    # so a log hyperparameter's values are recoded and its ticks relabelled.
    # See parallel_coordinates_plot.
    needs_config_space = True

    @classmethod
    def plot(cls, result, metric=None, config_space=None):
        return parallel_coordinates_plot(result, metric, config_space=config_space)


class PartialDependence(Figure):
    """One hyperparameter's partial dependence + ICE curves, picked via a
    dropdown — DeepCave's own small multiples become one Figure entry with a
    picker, matching every other multi-facet figure in this app.

    No `views`: which hyperparameter is showing isn't a small enumerable set
    of ways to look at the *same* data (contrast importance's game/rendering
    or performance's axis choices) — it changes what's being explained. And
    unlike the cube, it isn't cheap enough to precompute every option's data
    upfront: fitting a surrogate and predicting across a grid for every
    hyperparameter, on every page load, for a figure that only ever shows
    one at a time, would be pure waste. So, like local ablation
    (`hyperparameter_importance`'s "Local" game), this is fetched on demand
    — see the `partial_dependence` endpoint (ui/views.py) and
    experiment_detail.html's `refreshPartialDependence`. `plot()` is never
    called; it stays `None` (the base class default) so `plot_json` ships
    nothing for it and the client fetches everything, including the default
    hyperparameter shown on load.
    """

    key = "partial_dependence"
    label = _("Partial dependence (PDP/ICE)")
    home_tab = MISC
    shape = LONG
    per_metric = True
    # The whole figure is deferred, so the computation's name matches the key
    # here — unlike local ablation, which is one view of a figure whose other
    # views are precomputed.
    deferred = (("partial_dependence", _("Partial dependence (PDP/ICE)")),)


class AcquisitionSlice(Figure):
    """What the optimizer would look at next along one hyperparameter, and how a
    stated prior changes that.

    The acquisition function along that hyperparameter, weighted by the prior
    stated on `Prior` beside it. A prior does not move the surrogate's
    prediction — it multiplies the acquisition — so this curve is where its
    effect actually is. The two are separate figures so each can be placed and
    sized on its own; one script draws both, and one request feeds them.

    Deferred and fetched per (metric, hyperparameter), for partial dependence's
    reason exactly: it fits a surrogate and predicts across a grid, and a model
    with six hyperparameters would pay that six times on every page load for a
    panel that shows one.

    Three attributes deliberately unset. `needs_config_space` would do nothing,
    since `plot()` is never called here — the endpoint does that work itself, as
    `surrogate_uncertainty` does. `views` would be the wrong shape: which
    hyperparameter is shown is not another way of looking at the same data, the
    argument `PartialDependence` makes above. And `absolute_scale` would give
    the reader a toggle the script redraws over, since the page's scale toggle
    only ever relayouts `yaxis`.

    Unlike every other figure here, this one's drawing is owned by its own
    script rather than by experiment_detail.html — see
    `ui/static/ui/acquisition.js`. The prior is dragged, so its traces depend
    on state the server never sees; shipping expected improvement in Python as
    well would mean maintaining it twice.
    """

    key = "acquisition_slice"
    label = _("Acquisition function")
    home_tab = ACQUISITION
    shape = LONG
    default_columns = 4
    per_metric = True
    # The computation feeds `Prior` as well, and keeps its name: it is stored in
    # settings.
    deferred = (("acquisition_slice", _("Acquisition and priors")),)


class Prior(Figure):
    """A belief about where one hyperparameter's good values are, and what the
    model makes of it.

    Stated by a distribution and its parameters, or drawn freehand by dragging
    the curve, with a decay that weakens it as the run proceeds. It weights the
    acquisition on `AcquisitionSlice` and is drawn by the same script, from the
    same request — so it has no `plot()` and nothing deferred of its own.

    Not per metric: a belief is about the search space, and is the same
    whichever score is being read. And usable before any run, when it is the
    only figure the page has.
    """

    key = "prior"
    label = _("Prior")
    home_tab = ACQUISITION
    shape = LONG
    default_columns = 4


class LocalEffects(Figure):
    """Every sampled trial's local ablation as a beeswarm — the spread of each
    hyperparameter's effect, rather than one trial's or an average.

    Deferred, and the only new figure that is: it needs one ablation game per
    trial, which no amount of sharing makes free (43 ms each with the explainer
    shared, against 79 ms without). So it is fetched on request like partial
    dependence, capped by `local_effects_max_trials`, and its autocompute
    setting is off by default like every other deferred computation.
    """

    key = "local_effects"
    label = _("Local effects across trials")
    # One HyperSHAP ablation game per sampled trial: local explanation's game,
    # across the run rather than for one trial.
    home_tab = HYPERSHAP
    shape = LONG
    # Sampled, so most trials have no point here to click or to light up — see
    # `_selection_meta`, which is why a plot names its trials rather than
    # letting position imply them.
    selects_trials = True
    per_metric = True
    deferred = (("local_effects", _("Local effects across trials")),)


class TrialDuration(Figure):
    """One bar per trial. The same for every metric, so it is drawn once."""

    key = "trial_duration"
    home_tab = MISC
    # Drawn from the trials alone, so it can be updated mid-run. See Figure.live.
    live = True
    label = _("Trial duration")
    selects_trials = True

    @classmethod
    def plot(cls, result, metric=None):
        return trial_duration_plot(result)


class Trials(Figure):
    """Every trial as a sortable table — a column per metric needs the width."""

    key = "trials"
    label = _("Trials")
    home_tab = MISC
    shape = TABLE
    # On Overview from the start: it is the figure you look things up in while
    # reading a chart, so it belongs beside the charts rather than a tab away.
    pinned = True
    #: Rows to a page. A long run's table is one you scroll past rather than
    #: read, so it is paged; the page holds enough that a short run never sees a
    #: pager at all, and the reader can change it on the page. Declared here so
    #: the template's field and the script that reads it start from one number.
    page_size = 50
    # A table, not a plot: its rows carry the trial index themselves and the
    # highlight is a class, not a restyle.
    selects_trials = True


#: The default order of each tab, read per `Figure.home_tab`, and of Overview,
#: read among the pinned ones. A reader rearranges either from there — see
#: `ui/layout.py`.
#:
#: The two configuration panels first, then the chart everything else is read
#: against, then the interactions in the order they get harder (a grid, a
#: ranking, a shape, a list, a breakdown). `SelectedConfiguration` is in this
#: list for its settings checkbox and its per-metric panels; the page renders it
#: into the sidebar rather than a tab (`Figure.in_sidebar`), so its position
#: here is not a position on one.
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
    # Every trial at once, then one hyperparameter at a time, then one trial at
    # a time — the same run at three magnifications, in that order.
    ParallelCoordinates,
    PartialDependence,
    LocalExplanation,
    LocalEffects,
    AcquisitionSlice,
    Prior,
    # Under the acquisition function and its prior, on their tab: where the
    # search has been, beside where it would go next.
    ConfigurationCube,
    Trials,
)

FIGURES_BY_KEY = {figure.key: figure for figure in FIGURES}
