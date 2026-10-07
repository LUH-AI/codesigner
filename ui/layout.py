"""How an experiment page's figures are arranged: which tab, what order, how wide.

An arrangement is a dict, stored as JSON:

    {"version": 1,
     "tabs": {"overview":    [{"key": "performance_over_time", "w": 2, "c": 0}, ...],
              "hypershap":   [...], "acquisition": [...], "misc": [...]}}

Each tab is an ordered list of slots, and each slot names a figure, how many of
the grid's four columns it takes, and which column it starts in. A figure is on
its home tab (`Figure.home_tab`) always, and on Overview as well when it has
been pinned there; the two slots are independent, so it can be wide on one and
narrow on the other.

The site's default is `config/figure_layout.toml`, and nothing else: there is no
order implied by the catalog and no default stored in the database. The file is
required, and `check` — run at start-up as a system check — refuses one that is
missing, unreadable or does not place every figure.

Everything that reads or writes an arrangement goes through `normalize`. A
stored one outlives the catalog it was saved against — figures are added,
renamed, given a different shape — and `normalize` turns whatever was stored
into one that is valid now, taking anything it has to fill in from the default.
"""

import copy
import tomllib
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from .figures import FIGURES, FIGURES_BY_KEY, OVERVIEW, TABS

VERSION = 1

#: The file's table for the Data Handling page, and for a draft setup's own
#: Overview of it; everything outside the two is the dashboard's.
DATA_TABLE = "data"
SETUP_TABLE = "setup"

#: The site's default arrangement. Required; see the file for its format.
DEFAULT_PATH = Path(settings.BASE_DIR) / "config" / "figure_layout.toml"


class LayoutFileError(Exception):
    """`config/figure_layout.toml` is missing, unreadable, or wrong."""


def _arrangeable():
    """The figures a tab can hold. Not the sidebar's."""
    return [f for f in FIGURES if not f.in_sidebar]


def _column(value, w):
    """*value* as the column a figure *w* wide starts in on the four-column
    grid, or None: the first of a pair when two wide, the first when four."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value and value + w <= 4 and value % w == 0 else None


def _read_file(path):
    path = Path(path or DEFAULT_PATH)
    try:
        return path, tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LayoutFileError(f"{path} is missing. It is the site's default figure layout "
                              f"and is required.") from None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LayoutFileError(f"{path} cannot be read: {exc}") from None


def _read_tabs(raw, tab_names, by_key, arrangeable, problems, *, pinned=None, prefix=""):
    """One page's tabs from the file: `{tab: [{"key", "w", "c"}]}`, checked.

    Every panel in *arrangeable* must be on its home tab exactly once. The
    *pinned* tab, if the page has one, may hold any of them besides.
    """
    unknown = sorted(set(raw) - set(tab_names))
    if unknown:
        problems.append(f"no such tab: {', '.join(prefix + u for u in unknown)}")
    tabs = {}
    for tab in tab_names:
        entries = raw.get(tab)
        if not isinstance(entries, list):
            problems.append(f"[{prefix}{tab}] is missing or not a list")
            tabs[tab] = []
            continue
        slots, seen = [], set()
        for i, entry in enumerate(entries, 1):
            where = f"{prefix}{tab}, entry {i}"
            if not isinstance(entry, dict):
                problems.append(f"{where}: not a table")
                continue
            key = entry.get("figure")
            panel = by_key.get(key)
            if panel is None or getattr(panel, "in_sidebar", False):
                problems.append(f"{where}: no figure called {key!r}")
                continue
            if tab != pinned and panel.home_tab != tab:
                problems.append(f"{where}: {key} lives on {panel.home_tab}, not {tab}")
                continue
            if key in seen:
                problems.append(f"{where}: {key} is on {tab} twice")
                continue
            seen.add(key)
            w, column = entry.get("width"), entry.get("column")
            if w not in panel.columns():
                problems.append(f"{where}: {key} cannot be {w!r} wide "
                                f"(its shape allows {', '.join(map(str, panel.columns()))})")
                continue
            c = column - 1 if isinstance(column, int) and not isinstance(column, bool) else None
            if _column(c, w) is None:
                problems.append(f"{where}: {key}, {w} wide, cannot start in column {column!r}")
                continue
            slots.append({"key": key, "w": w, "c": c})
        tabs[tab] = slots
    for panel in arrangeable:
        if not any(s["key"] == panel.key for s in tabs.get(panel.home_tab, [])):
            problems.append(f"{panel.key} is not placed on its tab, {prefix}{panel.home_tab}")
    return tabs


def read_default(path=None) -> dict:
    """The dashboard's default arrangement in *path*, checked; raises
    `LayoutFileError`.

    The file counts columns from 1, as a person placing figures does; an
    arrangement counts them from 0.
    """
    path, raw = _read_file(path)
    problems = []
    dashboard = {k: v for k, v in raw.items() if k not in (DATA_TABLE, SETUP_TABLE)}
    tabs = _read_tabs(dashboard, TABS, FIGURES_BY_KEY, _arrangeable(), problems, pinned=OVERVIEW)
    if problems:
        raise LayoutFileError(f"{path}:\n  " + "\n  ".join(problems))
    return {"version": VERSION, "tabs": tabs}


def read_data_default(path=None) -> dict:
    """The Data Handling page's arrangement in *path* (its `[data]` table),
    checked; raises `LayoutFileError`. `{section: [{"key", "w", "c"}]}`."""
    from datahandling.panels import DATA_PANELS, DATA_PANELS_BY_KEY
    from datahandling.sections import SECTIONS

    path, raw = _read_file(path)
    table = raw.get(DATA_TABLE)
    if not isinstance(table, dict):
        raise LayoutFileError(f"{path}:\n  [{DATA_TABLE}] is missing or not a table")
    problems = []
    tabs = _read_tabs(table, [s.slug for s in SECTIONS], DATA_PANELS_BY_KEY, DATA_PANELS,
                      problems, prefix=f"{DATA_TABLE}.")
    if problems:
        raise LayoutFileError(f"{path}:\n  " + "\n  ".join(problems))
    return tabs


@lru_cache(maxsize=1)
def _default():
    return read_default()


def read_setup_default(path=None) -> list:
    """A draft's setup Overview in *path* (its `[setup]` table), checked; raises
    `LayoutFileError`. `[{"key", "w", "c"}]`: any Data Handling panel, as the
    dashboard's Overview holds figures whose home is another tab."""
    from datahandling.panels import DATA_PANELS_BY_KEY

    path, raw = _read_file(path)
    table = raw.get(SETUP_TABLE)
    if not isinstance(table, dict):
        raise LayoutFileError(f"{path}:\n  [{SETUP_TABLE}] is missing or not a table")
    problems = []
    tabs = _read_tabs(table, [OVERVIEW], DATA_PANELS_BY_KEY, (), problems,
                      pinned=OVERVIEW, prefix=f"{SETUP_TABLE}.")
    if problems:
        raise LayoutFileError(f"{path}:\n  " + "\n  ".join(problems))
    return tabs[OVERVIEW]


@lru_cache(maxsize=1)
def _data_default():
    return read_data_default()


@lru_cache(maxsize=1)
def _setup_default():
    return read_setup_default()


def data_tabs(exp, setup=False) -> list:
    """The Data Handling page's tabs for *exp*, as `placed` gives the
    dashboard's: each section with its slots, a slot being its panel, its width
    and its column. The site's arrangement, for every reader. With *setup*, a
    draft's: its Overview is the setup's own (`[setup]`)."""
    from datahandling.panels import DATA_PANELS_BY_KEY
    from datahandling.sections import SECTIONS

    arranged = dict(_data_default())
    if setup:
        arranged[SECTIONS[0].slug] = _setup_default()
    return [{"section": section,
             "slots": [{"figure": DATA_PANELS_BY_KEY[slot["key"]], "w": slot["w"],
                        "c": slot["c"]}
                       for slot in arranged[section.slug]
                       if DATA_PANELS_BY_KEY[slot["key"]].applies(exp.data)]}
            for section in SECTIONS]


def default_layout() -> dict:
    """The site's default arrangement: `config/figure_layout.toml`."""
    return copy.deepcopy(_default())


def check(app_configs=None, **kwargs):
    """The system check that makes the file required: the app will not start
    without a readable one that places every figure and every Data Handling
    panel."""
    from django.core.checks import Error
    try:
        read_default()
        read_data_default()
        read_setup_default()
    except LayoutFileError as exc:
        return [Error(str(exc), hint="Every figure in ui/figures/catalog.py, and every "
                                     "Data Handling panel in datahandling/sections.py, has "
                                     "to be placed on its home tab in this file.",
                      id="ui.E001")]
    return []


def _from_default(tab, key):
    """*key*'s slot on *tab* in the default, if it has one there."""
    return next((dict(s) for s in _default()["tabs"][tab] if s["key"] == key), None)


def _width(figure, value, tab) -> int:
    try:
        return figure.fit(int(value))
    except (TypeError, ValueError):
        fallback = _from_default(tab, figure.key) or _from_default(figure.home_tab, figure.key)
        return fallback["w"]


def normalize(layout) -> dict:
    """*layout*, made valid against the catalog as it is now.

    - anything that is not an arrangement is the default;
    - a slot naming no figure, a figure off its home tab, or a figure twice on
      one tab is dropped — Overview is the one tab that may hold any figure;
    - a width the figure's shape does not allow is the widest it allows that
      is no wider — three columns becomes two — and a width that is not a
      number is the default's;
    - a column (`c`, which of the four a figure starts in) that its width
      cannot start in is dropped, and the page finds it one;
    - a figure the stored arrangement has never heard of — added since it was
      saved — goes at the end of its home tab as the default has it, and on
      Overview too if the default pins it. "Never heard of" is read off the
      home tabs, which always hold every figure: a figure missing from
      Overview alone was unpinned, and stays unpinned.
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
            kept = {"key": figure.key, "w": _width(figure, slot.get("w"), tab)}
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
        tabs[figure.home_tab].append(_from_default(figure.home_tab, figure.key))
        pinned = _from_default(OVERVIEW, figure.key)
        if pinned and all(s["key"] != figure.key for s in tabs[OVERVIEW]):
            tabs[OVERVIEW].append(pinned)
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
# Three places, most particular first: this reader's arrangement of this
# experiment, this reader's default for any experiment, and the site's default
# file. Each is normalised as it is read, so a stale one is repaired rather
# than passed on.

def _reader(user):
    """The row a reader's arrangement is stored under: their account, or none
    on an install without accounts, where there is one reader."""
    return user if getattr(user, "is_authenticated", False) else None


def site_default() -> dict:
    """What a reader with no default of their own sees."""
    return default_layout()


def user_default(user) -> dict:
    """What an experiment *user* has not arranged opens on: their own default,
    or the site's."""
    from .models import DefaultFigureLayout
    row = DefaultFigureLayout.objects.filter(user=_reader(user)).first()
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
    """Make *layout* what every experiment *user* has not arranged opens on."""
    from .models import DefaultFigureLayout
    layout = normalize(layout)
    DefaultFigureLayout.objects.update_or_create(user=_reader(user), defaults={"layout": layout})
    return layout


def reset_user_default(user) -> dict:
    """Forget *user*'s default, so the site's applies again; returns that."""
    from .models import DefaultFigureLayout
    DefaultFigureLayout.objects.filter(user=_reader(user)).delete()
    return site_default()
