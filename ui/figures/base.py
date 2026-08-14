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


class Figure:
    """Base class for an experiment page's analytics panels."""

    # Django would otherwise *instantiate* the class when a template resolves it.
    do_not_call_in_templates = True

    #: Stable identifier. Names the template, the setting and the DOM id, and is
    #: stored in settings — so renaming one is a data migration.
    key = ""
    #: The heading shown on the panel, and the label of its settings checkbox.
    label = ""
    #: HALF or FULL (see above).
    width = HALF
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
    #: A Plotly relayout for this figure's "absolute" y-scale, which gives it
    #: the absolute/relative toggle in its toolbar. None means no toggle. For
    #: a figure with `views`, this is instead a dict keyed by view, since the
    #: sensible "absolute" range can differ per view (e.g. a linear 0-1 score
    #: vs. a log-scale error).
    absolute_scale = None

    def __init_subclass__(cls, **kwargs):
        """Derive everything that follows from the key, once per subclass."""
        super().__init_subclass__(**kwargs)
        if cls.key:
            cls.template = f"ui/figures/{cls.key}.html"
            cls.setting_key = f"show_{cls.key}"
            cls.dom_id = f"figure-{cls.key}"

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
        first/default); a figure with no views ignores the argument.
        """
        return None
