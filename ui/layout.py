"""An arrangement maps each tab to ordered slots `{"key", "w", "c"}`, giving the
figure, its width out of four grid columns, and its start column. A figure sits
on its home tab and may also be pinned to Overview with an independent slot.
Every read or write goes through `normalize`.
"""

import copy
import tomllib
from functools import lru_cache
from pathlib import Path

from experiments.models import account

from .figures import FIGURES, FIGURES_BY_KEY, OVERVIEW, TABS
from .figures.data_handling import DATA_PANELS, DATA_PANELS_BY_KEY, SECTIONS

VERSION = 1

#: The `[data]` and `[setup]` tables define Data Handling layouts, and every other
#: table is the dashboard's.
DATA_TABLE = "data"
SETUP_TABLE = "setup"

#: Site default arrangement; required.
DEFAULT_PATH = Path(__file__).with_name("figure_layout.toml")


class LayoutFileError(Exception):
    """`LayoutFileError` flags `ui/figure_layout.toml` errors."""


def _arrangeable():
    return [f for f in FIGURES if not f.in_sidebar]


def _column(value, w):
    """`_column` checks that a figure *w* wide can start at *value* on the
    four-column grid, and returns None otherwise."""
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
    """Every panel in *arrangeable* must sit on its home tab once, and the
    *pinned* tab may contain any panel too.
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


@lru_cache
def read_default(path=None) -> dict:
    """The file counts columns from 1, and an arrangement counts from 0.
    """
    path, raw = _read_file(path)
    problems = []
    dashboard = {k: v for k, v in raw.items() if k not in (DATA_TABLE, SETUP_TABLE)}
    tabs = _read_tabs(dashboard, TABS, FIGURES_BY_KEY, _arrangeable(), problems, pinned=OVERVIEW)
    if problems:
        raise LayoutFileError(f"{path}:\n  " + "\n  ".join(problems))
    return {"version": VERSION, "tabs": tabs}


def _read_table(path, name, tab_keys, arrangeable, pinned=None) -> dict:
    path, raw = _read_file(path)
    table = raw.get(name)
    if not isinstance(table, dict):
        raise LayoutFileError(f"{path}:\n  [{name}] is missing or not a table")
    problems = []
    tabs = _read_tabs(table, tab_keys, DATA_PANELS_BY_KEY, arrangeable, problems,
                      pinned=pinned, prefix=f"{name}.")
    if problems:
        raise LayoutFileError(f"{path}:\n  " + "\n  ".join(problems))
    return tabs


@lru_cache
def read_data_default(path=None) -> dict:
    return _read_table(path, DATA_TABLE, [s.slug for s in SECTIONS], DATA_PANELS)


@lru_cache
def read_setup_default(path=None) -> list:
    """The `[setup]` table may contain any Data Handling panel, like the
    dashboard's Overview."""
    return _read_table(path, SETUP_TABLE, [OVERVIEW], (), pinned=OVERVIEW)[OVERVIEW]


def data_tabs(exp, setup=False) -> list:
    """`data_tabs` shapes *exp*'s Data Handling tabs like `placed`, from the
    site arrangement for every reader. With *setup*, the Overview comes from
    `[setup]`."""
    arranged = dict(read_data_default())
    if setup:
        arranged[SECTIONS[0].slug] = read_setup_default()
    return [{"section": section,
             "slots": [{"figure": DATA_PANELS_BY_KEY[slot["key"]], "w": slot["w"],
                        "c": slot["c"]}
                       for slot in arranged[section.slug]
                       if DATA_PANELS_BY_KEY[slot["key"]].applies(exp.data)]}
            for section in SECTIONS]


def default_layout() -> dict:
    return copy.deepcopy(read_default())


def check(app_configs=None, **kwargs):
    """Django's system check fails unless the file places every figure and Data
    Handling panel."""
    from django.core.checks import Error
    try:
        read_default()
        read_data_default()
        read_setup_default()
    except LayoutFileError as exc:
        return [Error(str(exc), hint="Every figure in ui/figures/catalog.py, and every "
                                     "Data Handling panel in ui/figures/data_handling.py, has "
                                     "to be placed on its home tab in this file.",
                      id="ui.E001")]
    return []


def _from_default(tab, key):
    return next((dict(s) for s in read_default()["tabs"][tab] if s["key"] == key), None)


def _width(figure, value, tab) -> int:
    try:
        return figure.fit(int(value))
    except (TypeError, ValueError):
        fallback = _from_default(tab, figure.key) or _from_default(figure.home_tab, figure.key)
        assert fallback is not None
        return fallback["w"]


def normalize(layout) -> dict:
    """`normalize` drops slots naming no figure, off their home tab outside
    Overview, or repeated on a tab. An unallowed width shrinks to the widest
    allowed, and an invalid column is dropped for the page to choose. A figure
    absent from every home tab is new and goes where the default puts it, while
    a figure absent only from Overview stays unpinned.
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
            figure = FIGURES_BY_KEY.get(slot.get("key", ""))
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
    """`placed` skips a figure switched off in *shown* but keeps the figure's
    slot in the arrangement.
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


# Sources: this experiment's arrangement, the reader's default, the site file.

def user_default(user) -> dict:
    """`user_default` returns *user*'s own default, else the site's."""
    from .models import DefaultFigureLayout
    row = DefaultFigureLayout.objects.filter(user=account(user)).first()
    return normalize(row.layout) if row else default_layout()


def layout_for(experiment, user) -> dict:
    from .models import FigureLayout
    row = FigureLayout.objects.filter(experiment=experiment, user=account(user)).first()
    return normalize(row.layout) if row else user_default(user)


def save(experiment, user, layout) -> dict:
    from .models import FigureLayout
    layout = normalize(layout)
    FigureLayout.objects.update_or_create(experiment=experiment, user=account(user),
                                          defaults={"layout": layout})
    return layout


def reset(experiment, user) -> dict:
    """`reset` drops the arrangement and returns *user*'s default."""
    from .models import FigureLayout
    FigureLayout.objects.filter(experiment=experiment, user=account(user)).delete()
    return user_default(user)


def save_user_default(user, layout) -> dict:
    """`save_user_default` writes *user*'s default arrangement."""
    from .models import DefaultFigureLayout
    layout = normalize(layout)
    DefaultFigureLayout.objects.update_or_create(user=account(user), defaults={"layout": layout})
    return layout


def reset_user_default(user) -> dict:
    """`reset_user_default` drops *user*'s default for the site's."""
    from .models import DefaultFigureLayout
    DefaultFigureLayout.objects.filter(user=account(user)).delete()
    return default_layout()
