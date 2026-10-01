"""How an experiment page's figures are arranged: which tab, what order, how wide.

An arrangement is a dict, stored as JSON:

    {"version": 1,
     "tabs": {"overview":    [{"key": "performance_over_time", "w": 2}, ...],
              "hypershap":   [...], "acquisition": [...], "misc": [...]}}

Each tab is an ordered list of slots, and each slot names a figure and how many
of the grid's four columns it takes. A figure is on its home tab
(`Figure.home_tab`) always, and on Overview as well when it has been pinned
there; the two slots are independent, so it can be wide on one and narrow on the
other.

Everything that reads or writes one goes through `normalize`. A stored
arrangement outlives the catalog it was saved against — figures are added,
renamed, given a different shape — and `normalize` is what turns whatever was
stored into one that is valid against the catalog now, so nothing downstream has
to wonder.
"""

from .figures import FIGURES, FIGURES_BY_KEY, OVERVIEW, TABS

VERSION = 1


def _arrangeable():
    """The figures a tab can hold, in catalog order. Not the sidebar's."""
    return [f for f in FIGURES if not f.in_sidebar]


def default_layout() -> dict:
    """Each figure on its home tab at its opening width, in catalog order, and
    the pinned ones on Overview as well."""
    figures = _arrangeable()
    tabs = {tab: [] for tab in TABS}
    for figure in figures:
        slot = {"key": figure.key, "w": figure.opening_columns()}
        tabs[figure.home_tab].append(dict(slot))
        if figure.pinned:
            tabs[OVERVIEW].append(dict(slot))
    return {"version": VERSION, "tabs": tabs}


def _width(figure, value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return figure.opening_columns()
    return figure.fit(value)


def _column(value, w):
    """*value* as the column a figure *w* wide starts in on the four-column
    grid, or None: the first of a pair when two wide, the first when four."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value and value + w <= 4 and value % w == 0 else None


def normalize(layout) -> dict:
    """*layout*, made valid against the catalog as it is now.

    - anything that is not an arrangement is the default;
    - a slot naming no figure, a figure off its home tab, or a figure twice on
      one tab is dropped — Overview is the one tab that may hold any figure;
    - a width the figure's shape does not allow is the widest it allows that
      is no wider — three columns becomes two;
    - a column (`c`, which of the four a figure starts in) that its width
      cannot start in is dropped, and the page finds it one;
    - a figure the stored arrangement has never heard of — added to the catalog
      since it was saved — goes at the end of its home tab at its opening
      width, and on Overview too if it is pinned by default. "Never heard of"
      is read off the home tabs, which always hold every figure: a figure
      missing from Overview alone was unpinned, and stays unpinned.
    """
    if not isinstance(layout, dict) or not isinstance(layout.get("tabs"), dict):
        return default_layout()

    stored = layout["tabs"]
    tabs, known = {}, set()
    for tab in TABS:
        slots, seen = [], set()
        for slot in stored.get(tab) or []:
            if not isinstance(slot, dict):
                continue
            figure = FIGURES_BY_KEY.get(slot.get("key"))
            if figure is None or figure.in_sidebar or figure.key in seen:
                continue
            if tab != OVERVIEW and figure.home_tab != tab:
                continue
            seen.add(figure.key)
            kept = {"key": figure.key, "w": _width(figure, slot.get("w"))}
            column = _column(slot.get("c"), kept["w"])
            if column is not None:
                kept["c"] = column
            slots.append(kept)
        tabs[tab] = slots
        if tab != OVERVIEW:
            known |= seen

    for figure in _arrangeable():
        if figure.key in known:
            continue
        tabs[figure.home_tab].append({"key": figure.key, "w": figure.opening_columns()})
        if figure.pinned and all(s["key"] != figure.key for s in tabs[OVERVIEW]):
            tabs[OVERVIEW].append({"key": figure.key, "w": figure.opening_columns()})
    return {"version": VERSION, "tabs": tabs}


def placed(layout, shown) -> list:
    """The tabs as the page renders them: each a list of slots, a slot being
    its figure, its width and the widths it may be.

    *shown* is the figures this experiment shows (its `show_<key>` settings).
    One that is switched off is skipped here but kept in the arrangement, so
    switching it back on puts it back where it was.
    """
    shown = {f.key for f in shown}
    layout = normalize(layout)
    tabs = []
    for tab in TABS:
        slots = []
        for slot in layout["tabs"][tab]:
            if slot["key"] not in shown:
                continue
            figure = FIGURES_BY_KEY[slot["key"]]
            slots.append({"figure": figure, "w": slot["w"], "c": slot.get("c"),
                          "widths": " ".join(str(w) for w in figure.columns())})
        tabs.append({"tab": tab, "slots": slots})
    return tabs


# ── where an arrangement comes from ──────────────────────────────────────────
#
# Four places, most particular first: this reader's arrangement of this
# experiment, this reader's default for any experiment, the site's default,
# and the catalog's own order. Each is normalised as it is read, so a stale one
# at any level is repaired rather than passed on.

def _reader(user):
    """The row a reader's arrangement is stored under: their account, or none
    on an install without accounts, where there is one reader."""
    return user if getattr(user, "is_authenticated", False) else None


def site_default() -> dict:
    """What a reader with no default of their own sees: the site's, if one was
    set, or the catalog's own."""
    from .models import GlobalSettings
    stored = GlobalSettings.get_solo().default_figure_layout
    return normalize(stored) if stored else default_layout()


def user_default(user) -> dict:
    """What an experiment *user* has not arranged opens on: their own default,
    or the site's. Without accounts the one reader's default is the site's."""
    from .models import DefaultFigureLayout
    reader = _reader(user)
    row = DefaultFigureLayout.objects.filter(user=reader).first() if reader else None
    return normalize(row.layout) if row else site_default()


def layout_for(experiment, user) -> dict:
    """*user*'s arrangement of *experiment*, falling back to their default."""
    from .models import FigureLayout
    row = FigureLayout.objects.filter(experiment=experiment, user=_reader(user)).first()
    return normalize(row.layout) if row else user_default(user)


def save(experiment, user, layout) -> dict:
    """Keep *layout* as *user*'s for *experiment*, normalised; returns it."""
    from .models import FigureLayout
    layout = normalize(layout)
    FigureLayout.objects.update_or_create(experiment=experiment, user=_reader(user),
                                          defaults={"layout": layout})
    return layout


def reset(experiment, user) -> dict:
    """Forget *user*'s arrangement of *experiment*, so their default applies
    again; returns that."""
    from .models import FigureLayout
    FigureLayout.objects.filter(experiment=experiment, user=_reader(user)).delete()
    return user_default(user)


def save_user_default(user, layout) -> dict:
    """Make *layout* what every experiment *user* has not arranged opens on.

    Without accounts that is every experiment anybody has not arranged, so it
    is the site's default.
    """
    from .models import DefaultFigureLayout, GlobalSettings
    layout = normalize(layout)
    reader = _reader(user)
    if reader is None:
        settings = GlobalSettings.get_solo()
        settings.default_figure_layout = layout
        settings.save(update_fields=["default_figure_layout"])
    else:
        DefaultFigureLayout.objects.update_or_create(user=reader, defaults={"layout": layout})
    return layout


def reset_user_default(user) -> dict:
    """Forget *user*'s default, so the site's applies again; returns that.

    Without accounts the default *is* the site's, so this goes back to the
    catalog's own order.
    """
    from .models import DefaultFigureLayout, GlobalSettings
    reader = _reader(user)
    if reader is None:
        settings = GlobalSettings.get_solo()
        settings.default_figure_layout = {}
        settings.save(update_fields=["default_figure_layout"])
    else:
        DefaultFigureLayout.objects.filter(user=reader).delete()
    return user_default(user)
