"""What a figure is.

A **figure** is one analytics panel on an experiment page. Subclass `Figure` to
declare a new one; the class *is* the declaration — its attributes say how the
figure is named, how wide it sits, whether it is per-metric, and how it draws.
Nothing outside the class needs editing: the visibility setting, its checkbox on
the settings pages, its slot on the page and the id its plot is drawn into are
all derived from `key`.

    class TrialDuration(Figure):
        key = "trial_duration"
        label = _("Trial duration")
        width = HALF

        @classmethod
        def plot(cls, result, metric=None):
            return trial_duration_plot(result)

Then add it to `FIGURES` in `catalog.py` (order there is page order) and write
`ui/templates/ui/figures/trial_duration.html`.
"""

#: Widths a figure can take. HALF figures tile two-up in reading order; FULL
#: ones span the page below them (a table with a column per metric needs it).
HALF = "half"
FULL = "full"
#: The whole content area, across the grid *and* the column beside it. FULL
#: spans the grid's columns, which on a wide window is only the left pane — the
#: trials table has the right one. A figure that needs the page's whole width is
#: rendered under both rather than inside either.
PAGE = "page"

#: And heights. SINGLE is one row of the grid; DOUBLE is two, for a figure whose
#: content is square rather than wide — a scatter over a projected space reads as
#: a strip at one row and as a plane at two.
SINGLE = "single"
DOUBLE = "double"


class Figure:
    """Base class for an experiment page's analytics panels."""

    # Django would otherwise *instantiate* the class when a template resolves it.
    do_not_call_in_templates = True

    #: Stable identifier. Names the template, the setting and the DOM id, and is
    #: stored in settings — so renaming one is a data migration.
    key = ""
    #: The heading shown on the panel, and the label of its settings checkbox.
    label = ""
    #: HALF, FULL or PAGE (see above).
    width = HALF
    #: SINGLE or DOUBLE (see above). Height is declared separately from width
    #: because they answer different questions: how much of the page a figure
    #: needs beside it, and how much it needs under it.
    height = SINGLE
    #: True when the figure is drawn per evaluation metric, so it is rebuilt
    #: when the metric selector changes; False when one plot covers the run.
    per_metric = False
    #: Named view keys this figure offers, in declaration order; the first is
    #: shown until the switcher is touched. Empty (the default) means no
    #: switcher — every figure that predates this renders exactly as before.
    #: A figure with views owns its own selector markup in its template (a
    #: flat dropdown, two composed axis pickers, whatever fits it), since the
    #: shapes differ enough that forcing one generic widget would fit neither
    #: well; only the underlying draw/lookup plumbing is shared (see
    #: experiment_detail.html's `payloadFor`).
    views = ()
    #: Which of `views` a figure opens on, when that is not the first one
    #: declared. Separate from the order of `views`, because the order they read
    #: in and the one worth showing first are different questions — a figure can
    #: list its views from simplest to richest and still open on the richest.
    default_view = ""
    #: A Plotly relayout for this figure's "absolute" y-scale, which gives it
    #: the absolute/relative toggle in its toolbar. None means no toggle. For
    #: a figure with `views`, this is instead a dict keyed by view, since the
    #: sensible "absolute" range can differ per view (e.g. a linear 0-1 score
    #: vs. a log-scale error).
    absolute_scale = None
    #: Computations this figure fetches on demand rather than shipping in the
    #: page payload, as `(name, label)` pairs. Each gets its own
    #: `autocompute_<name>` setting: on (the default) fetches as soon as the
    #: figure needs it, off puts a "Compute" button there instead, so a page
    #: reload costs nothing until asked. Empty means the figure has nothing
    #: deferred and gains no such setting.
    #:
    #: The names are the *computation's*, not the figure's, because one of them
    #: isn't a whole figure: local ablation is a **view of** the importance
    #: figure, so `autocompute_hyperparameter_importance` would claim the
    #: importance numbers are deferred when they are computed once at run
    #: completion and stored.
    deferred = ()
    #: True when this figure's `plot()` takes a `config_space` keyword.
    #: A builder that draws hyperparameter *values* on an axis needs to know how
    #: each one is scaled — a log hyperparameter drawn on a linear axis crushes
    #: the region the search actually explored into a stripe — and the stored
    #: result carries values, not the space they came from. Declared rather than
    #: passed to everything, so the figures that don't need it keep the plain
    #: `(result, metric)` signature they have always had.
    needs_config_space = False
    #: True when this figure draws every trial, and so can both take a click
    #: naming one and show which one is selected. One trial is selected at a
    #: time across the whole page, so these agree with each other and with the
    #: Selected configuration panel — see experiment_detail.html's selection
    #: bus, and `_selection_meta` in plots.py for how a plot says where its
    #: trials are. A figure that draws a summary rather than the trials
    #: themselves (importance, interactions, partial dependence) has no point to
    #: click and nothing to highlight.
    selects_trials = False
    #: True when this figure belongs in the page's sidebar rather than in the
    #: figure grid. For a figure that answers a click made anywhere on the page
    #: rather than showing something of its own: in the grid it scrolls out of
    #: view exactly while it is being used. Everything else about such a figure
    #: — its visibility setting, its per-metric panels, its place in the
    #: catalog — is unchanged; only where the page puts it.
    in_sidebar = False
    #: True when this figure can be redrawn from the trials alone, and so can be
    #: updated while a run is still going. Trial performance, trial duration, the
    #: projection and parallel coordinates all read nothing but `result.trials`;
    #: importance, interactions, partial dependence and local effects each need a
    #: surrogate fit or 2^n_hp coalition evaluations, and none of those may run
    #: per trial. A partial result leaves their fields empty, which every one of
    #: them already handles — it is what a cancelled run produces.
    #:
    #: About *plot* payloads only. A figure rendered as server-side HTML is not
    #: redrawn by swapping Plotly JSON, so it does not set this even when it
    #: does update live: the trials table takes rendered rows through the same
    #: poll and appends them (see `ui/views.py`'s `_live_payloads`), and the two
    #: configuration panels do not update at all. See `run_status`.
    live = False
    #: True when this figure moves into a column of its own beside the grid once
    #: the window is wide enough for one. For a figure that is read *against*
    #: the others rather than in sequence with them — the trials table, which
    #: you look things up in while reading a chart. It has no effect on a narrow
    #: window, where it simply renders where its catalog order puts it.
    in_side_column = False

    def __init_subclass__(cls, **kwargs):
        """Derive everything that follows from the key, once per subclass."""
        super().__init_subclass__(**kwargs)
        if cls.key:
            cls.template = f"ui/figures/{cls.key}.html"
            cls.setting_key = f"show_{cls.key}"
            cls.dom_id = f"figure-{cls.key}"

    @classmethod
    def opening_view(cls):
        """The view this figure draws before anything is switched."""
        return cls.default_view or (cls.views[0] if cls.views else None)

    @classmethod
    def absolute_scale_for(cls, metric):
        """This figure's absolute scale under *metric*, or None for no toggle.

        The declaration above is the answer for every figure whose axes mean the
        same thing whatever is being measured. A figure drawing the *score* on
        an axis has to resolve it instead: "absolute" means the metric's own
        full range, and an unbounded metric has none — so it overrides this and
        returns None for those, which hides the toggle rather than pinning the
        axis to a range invented for a different metric.
        """
        return cls.absolute_scale

    #: Set by __init_subclass__; declared here so the base class is usable too.
    template = ""
    setting_key = ""
    dom_id = ""

    @classmethod
    def plot(cls, result, metric=None, view=None):
        """This figure's Plotly plot, or None if it draws a table instead.

        Figures with `per_metric = True` are called once per metric; the rest
        are called once with `metric=None`. A figure with `views` is called
        once per declared view (`view` is one of `views`; None means the
        first/default); a figure with no views ignores the argument. A figure
        declaring `needs_config_space` also takes `config_space`, which is None
        when the model isn't available to build one from (a custom model viewed
        read-only).
        """
        return None


def autocompute_key(name: str) -> str:
    """The settings key for a deferred computation named *name*.

    A boolean, not a tri-state: "never compute this" is already expressible as
    switching the figure off (`show_<key>`), which suppresses its fetch too. Two
    states means `_posted_settings`' `bool()` coercion needs no special case and
    no new widget type is needed.
    """
    return f"autocompute_{name}"


def deferred_computations():
    """Every `(name, label)` any figure defers, in catalog order.

    Imported lazily to avoid a cycle: `catalog` imports from this module.
    """
    from .catalog import FIGURES
    return [pair for figure in FIGURES for pair in figure.deferred]
