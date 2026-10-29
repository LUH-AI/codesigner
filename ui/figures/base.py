"""A `Figure` subclass declares one panel, and its visibility setting, checkbox,
slot and DOM id all derive from `key`. A new figure also needs an entry in
`FIGURES` in `catalog.py` and a template at
`ui/templates/ui/figures/<key>.html`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: The tab bar shows the tabs in order, and each figure's home is one of the
#: last three. Overview contains pinned figures.
OVERVIEW = "overview"
HYPERSHAP = "hypershap"
ACQUISITION = "acquisition"
MISC = "misc"
TABS = (OVERVIEW, HYPERSHAP, ACQUISITION, MISC)

#: Each shape allows widths of one, two or four grid columns, and `app.css` sets
#: its height per width. SQUAT tiles grow to about a screen tall at full width.
#: LONG plots read along an axis and start at two columns. STACK fits its
#: content, and TABLE scrolls inside a fixed block.
SQUAT = "squat"
LONG = "long"
STACK = "stack"
TABLE = "table"
#: A card contains Data Handling controls or text at its natural height.
CARD = "card"

#: shape -> allowed widths, narrowest first.
SHAPES = {
    SQUAT: (1, 2, 4),
    LONG: (2, 4),
    STACK: (2, 4),
    TABLE: (2,),
    CARD: (1, 2, 4),
}


#: An experiment's pages: the dashboard and Data Handling.
DASHBOARD = "dashboard"
DATA = "data"


class Panel:
    """Every block gets the same slot, shapes and placement, whether it is a
    `Figure` or a `data_handling.DataPanel`. Placement comes from
    `ui/figure_layout.toml`.
    """

    #: Stops Django instantiating the class in templates.
    do_not_call_in_templates = True

    #: Renaming a key is a data migration, since the key names the template, DOM
    #: id and stored setting.
    key = ""
    #: Panel heading.
    label = ""
    #: DASHBOARD or DATA.
    page = DASHBOARD
    #: Home tab; on the dashboard, never Overview.
    home_tab = MISC
    #: The shape is SQUAT, LONG, STACK, TABLE or CARD, and
    #: `ui/figure_layout.toml` decides placement and opening width.
    shape = SQUAT
    #: Template directory.
    template_dir = "ui/figures"
    #: True for panels meaningful only for a forecast; see `applies`.
    forecast_only = False

    #: Set by `__init_subclass__`.
    template = ""
    dom_id = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.key:
            cls.template = f"{cls.template_dir}/{cls.key}.html"
            cls.dom_id = f"figure-{cls.key}"

    @classmethod
    def applies(cls, data) -> bool:
        """`applies` accepts the panel for *data*'s page."""
        return not cls.forecast_only or data.forecasts

    @classmethod
    def columns(cls):
        """`columns` lists allowed widths, narrowest first."""
        return SHAPES[cls.shape]

    @classmethod
    def fit(cls, columns):
        """`fit` picks the widest allowed width within *columns*, else the
        narrowest."""
        allowed = cls.columns()
        fitting = [w for w in allowed if w <= columns]
        return fitting[-1] if fitting else allowed[0]


class Figure(Panel):
    """`Figure` extends `Panel` to trial plots that `show_<key>` switches."""
    #: Per-metric figures rebuild when the metric selector changes.
    per_metric = False
    #: The first named view shows by default, and an empty tuple means no
    #: switcher. A figure with views writes its own selector markup and shares
    #: only the draw plumbing (`payloadFor`).
    views: tuple[str, ...] = ()
    #: `default_view` overrides the first view as the opening one.
    default_view = ""
    #: The "absolute" y-scale toggle takes a Plotly relayout, or a dict of
    #: relayouts keyed by view. None means no toggle.
    absolute_scale: dict | None = None
    #: Each `(name, label)` computation is fetched on demand and has an
    #: `autocompute_<name>` setting. The names follow the computation, since
    #: local ablation is a view of the importance figure.
    deferred = ()
    #: Figures setting `needs_config_space` get `config_space` in `plot()` to
    #: draw log-scaled values correctly.
    needs_config_space = False
    #: A figure that draws every trial takes clicks and shows the page-wide
    #: selection.
    selects_trials = False
    #: A HyperSHAP game figure gets its other baselines from `hypershap_figures`
    #: on request.
    explains_games = False
    #: True for a figure that answers clicks elsewhere, since in the grid it
    #: would scroll away while in use.
    in_sidebar = False
    #: Live figures redraw from the trials alone during a run. Surrogate-backed
    #: figures cannot run per trial, and the trials table updates through
    #: `_live_payloads` instead.
    live = False
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.key:
            cls.setting_key = f"show_{cls.key}"

    @classmethod
    def opening_view(cls):
        """`opening_view` is drawn before any switch."""
        return cls.default_view or (cls.views[0] if cls.views else None)

    @classmethod
    def absolute_scale_for(cls, metric) -> dict | None:
        """A figure that plots the score overrides this to return None for an
        unbounded metric, hiding the toggle.
        """
        return cls.absolute_scale

    #: Set by `__init_subclass__`.
    setting_key = ""

    @classmethod
    def plot(cls, result, metric=None, view=None) -> go.Figure | None:
        """Per-metric figures get one call per metric, and view figures one per
        view. `config_space` is None when no model can build one.
        """
        return None


def autocompute_key(name: str) -> str:
    """Two states suffice, since switching the figure off already means never
    computing.
    """
    return f"autocompute_{name}"


def deferred_computations():
    """`deferred_computations` defers importing `catalog`, which imports `base`.
    """
    from .catalog import FIGURES
    return [pair for figure in FIGURES for pair in figure.deferred]
