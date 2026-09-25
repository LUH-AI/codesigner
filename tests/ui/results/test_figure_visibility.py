"""Which figures an experiment page shows is a setting.

A "figure" is one analytics panel on an experiment page. The eleven of them are
declared once in `ui.figures.catalog`; each has a visibility setting, on by
default, editable on the default-experiment-settings page. Turning one off must
remove it from the page without disturbing the rest — the panels share a script,
so a hidden figure is exactly where a stale DOM lookup would break the others.

Performance-over-trials and error-over-time used to be two separate figures;
they are now one ("performance_over_time") with four views (trial/time x
score/error axes) — see test_plots.py for the view builder itself and
test_figures_view.py for the per-view JSON shape this collapses into.
"""

from django.urls import reverse

from ui.figures import (
    FIGURES, FIGURES_BY_KEY, FULL, HALF, HP_GAME_FIELDS, PAGE, Figure,
)
from ui.models import Experiment, GlobalSettings

EXPECTED_KEYS = [
    # Rendered into the sidebar rather than the grid (Figure.in_sidebar), but
    # still a figure with a visibility setting like any other.
    "selected_configuration",
    "best_configuration",
    "hyperparameter_importance",
    "performance_over_time",
    # Directly under it: the same run read along its other axis.
    "configuration_cube",
    # Five readings of one computation, each its own figure so they can be read
    # side by side and switched off one at a time.
    "interactions_heatmap",
    "interactions_top_pairs",
    "interactions_graph",
    "interactions_coalitions",
    "interactions_orders",
    "trial_duration",
    # Every trial at once, then one hyperparameter at a time, then one trial at
    # a time.
    "parallel_coordinates",
    "partial_dependence",
    # Beside it: the same slice through the same surrogate, read forwards.
    # The ablation game, as its own figure: the other three games explain the
    # search, this explains one trial, and it has no interactions to give the
    # figures that read them.
    "local_explanation",
    "local_effects",
    "acquisition_slice",
    "trials",
]


def _experiment():
    """An experiment with one finished trial, so every figure has data."""
    return Experiment.objects.create(
        name="figures", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], current_metric="accuracy",
        original_metric="accuracy", seed=0,
        result={
            "stats": {"submitted": 1, "finished": 1, "running": 0},
            "data": [{"config_id": 1, "cost": 0.2, "time": 2.5,
                      "scores": {"accuracy": 0.8, "f1": 0.7},
                      "incumbent_score": 0.8, "incumbent_config_id": 1}],
            "configs": {"1": {"n_estimators": 100}},
            "config_origins": {"1": "Random Search"}, "optimizer_state": {},
            "primary_metric": "accuracy", "best_score": 0.8, "best_config_id": "1",
            "hyperparameter_importance": {"accuracy": {"n_estimators": 1.0}},
            "hyperparameter_importance_warning": {}, "trials_limit": None,
        },
    )


def _page(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()


def _hide(*keys):
    gs = GlobalSettings.get_solo()
    gs.default_experiment_settings = {f"show_{k}": False for k in keys}
    gs.save(update_fields=["default_experiment_settings"])


def test_catalog_declares_every_figure_in_page_order():
    """The catalog is the one list defining what a figure is; everything else
    (settings keys, the settings page, the detail page) reads it."""
    assert [c.key for c in FIGURES] == EXPECTED_KEYS


def test_every_figure_has_a_label_and_a_template():
    for figure in FIGURES:
        assert str(figure.label)
        assert figure.template == f"ui/figures/{figure.key}.html"
        assert figure.setting_key == f"show_{figure.key}"


def test_declaring_a_subclass_derives_everything_from_its_key():
    """Adding a figure is subclassing it: the template it renders, the setting
    that hides it and the id its plot mounts in all follow from the key, so a
    new figure needs no wiring anywhere else."""
    class Whatever(Figure):
        key = "whatever"
        label = "Whatever"

    assert Whatever.template == "ui/figures/whatever.html"
    assert Whatever.setting_key == "show_whatever"
    assert Whatever.dom_id == "figure-whatever"
    # unspecified display characteristics fall back to the common case
    assert Whatever.width == HALF
    assert Whatever.per_metric is False
    assert Whatever.absolute_scale is None
    assert Whatever.plot(result=None) is None


def test_each_figure_declares_its_width():
    """Width is declared per figure, and is one of the three supported layouts —
    a half tile, a span of the grid, or the whole content area.

    FULL and PAGE are not the same span once the window is wide enough for two
    panes: FULL spans the grid, which is then only the left one, while PAGE
    spans the grid *and* the column beside it."""
    assert {f.width for f in FIGURES} <= {HALF, FULL, PAGE}
    assert FIGURES_BY_KEY["acquisition_slice"].width == PAGE
    assert [f.key for f in FIGURES if f.width == PAGE] == ["acquisition_slice"]
    assert FIGURES_BY_KEY["trials"].width == FULL
    assert FIGURES_BY_KEY["trial_duration"].width == HALF
    # Click-to-select drives the selected-configuration panel, so the chart you
    # click wants the width and the adjacency.
    assert FIGURES_BY_KEY["performance_over_time"].width == FULL
    # Axis pickers (and a potential 3D plot) need the room a half-tile lacks.
    assert FIGURES_BY_KEY["configuration_cube"].width == FULL
    # One line per trial across every hyperparameter needs the same room.
    assert FIGURES_BY_KEY["parallel_coordinates"].width == FULL
    # ICE lines for a whole trial history need the same room too.
    assert FIGURES_BY_KEY["partial_dependence"].width == FULL


def test_only_metric_dependent_figures_are_marked_per_metric():
    """Per-metric figures are rebuilt when the metric changes; trial duration
    is the same plot for every metric, and the tables aren't plots at all."""
    per_metric = {f.key for f in FIGURES if f.per_metric}
    assert per_metric == {
        "hyperparameter_importance",
        "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
        "interactions_coalitions", "interactions_orders",
        "performance_over_time", "configuration_cube", "parallel_coordinates",
        "partial_dependence", "acquisition_slice", "local_explanation",
        "local_effects",
    }


def test_the_scale_toggle_is_declared_not_hardcoded():
    """A figure opts into the absolute/relative y-scale button by declaring the
    range its 'absolute' means; the page builds the button from that.

    performance_over_time declares one range per view, since which one is
    "absolute" depends on whether the score or the error axis is showing.
    """
    scale = FIGURES_BY_KEY["performance_over_time"].absolute_scale
    assert scale["trial-score"] == {"yaxis.range": [0, 1], "yaxis.autorange": False}
    assert scale["time-score"] == {"yaxis.range": [0, 1], "yaxis.autorange": False}
    # the error views' y-axis is log, so their range is in log10 units
    assert scale["trial-error"] == {"yaxis.range": [-3, 0], "yaxis.autorange": False}
    assert scale["time-error"] == {"yaxis.range": [-3, 0], "yaxis.autorange": False}
    assert FIGURES_BY_KEY["trials"].absolute_scale is None


def test_all_figures_show_by_default(client):
    html = _page(client, _experiment())
    for figure in FIGURES:
        assert f'data-figure="{figure.key}"' in html, figure.key


def test_unchecking_a_figure_removes_it_from_the_page(client):
    _hide("hyperparameter_importance")
    html = _page(client, _experiment())

    assert 'data-figure="hyperparameter_importance"' not in html
    # the others are untouched
    assert 'data-figure="performance_over_time"' in html
    assert 'data-figure="trials"' in html


def test_hiding_the_performance_figure_keeps_the_page_working(client):
    """The performance figure owns click-to-select, so the script reaches for it
    by id. Hidden, the page must still render its remaining figures."""
    _hide("performance_over_time")
    html = _page(client, _experiment())

    assert 'data-figure="performance_over_time"' not in html
    assert 'id="figure-performance_over_time"' not in html
    assert 'data-figure="best_configuration"' in html
    assert 'data-figure="trial_duration"' in html


def test_all_figures_can_be_hidden_at_once(client):
    _hide(*EXPECTED_KEYS)
    html = _page(client, _experiment())

    for figure in FIGURES:
        assert f'data-figure="{figure.key}"' not in html, figure.key
    assert "Trials" not in html.split("<main", 1)[-1]


def test_a_hidden_figure_ships_no_plot_data(client):
    """A figure that is off is not computed either — the JSON exists only to be
    drawn, so an absent figure should leave nothing behind in it."""
    _hide("trial_duration", "performance_over_time")
    html = _page(client, _experiment())
    static = html.split('id="static-plots-data"', 1)[1].split("</script>", 1)[0]
    per_metric = html.split('id="metric-plots-data"', 1)[1].split("</script>", 1)[0]

    assert "trial_duration" not in static
    assert "performance_over_time" not in per_metric
    # the figure still on the page is unaffected
    assert "hyperparameter_importance" in per_metric


def test_the_selected_configuration_panel_is_not_in_the_grid(client):
    """It answers a click made anywhere on the page, so it lives in the sidebar
    (Figure.in_sidebar) where it stays readable while the figures are being
    clicked. It is still a figure — still in the catalog, still with a
    visibility setting — only somewhere else."""
    html = _page(client, _experiment())
    grid = html.split('class="grid-2x2"', 1)[1]
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert FIGURES_BY_KEY["selected_configuration"].in_sidebar
    assert [f.key for f in FIGURES if f.in_sidebar] == ["selected_configuration"]
    assert 'data-figure="selected_configuration"' in sidebar
    assert 'data-figure="selected_configuration"' not in grid
    assert 'data-figure="best_configuration"' in grid, "the other panel stays"


def test_trial_performance_begins_a_row_of_its_own(client):
    """The chart every other figure is read against gets the full width of the
    grid, whatever the grid's column count is — and because a spanning item
    always starts a fresh row, it lands below the tiles ahead of it however
    many of them are switched off."""
    html = _page(client, _experiment())
    grid = html.split('class="grid-2x2"', 1)[1]

    best = grid.index('data-figure="best_configuration"')
    performance = grid.index('data-figure="performance_over_time"')

    assert best < performance
    slot = grid.rindex('<div class="slot', 0, performance)
    assert "wide" in grid[slot:performance], "spans every column"


def test_hiding_the_figure_above_it_keeps_it_spanning(client):
    """The case an ordering would get wrong: with the tile above it gone the
    grid reflows, and only the spanning row still behaves."""
    _hide("best_configuration")
    html = _page(client, _experiment())
    grid = html.split('class="grid-2x2"', 1)[1]

    performance = grid.index('data-figure="performance_over_time"')
    slot = grid.rindex('<div class="slot', 0, performance)

    assert "wide" in grid[slot:performance]


def test_a_figures_pickers_sit_in_one_row_above_it(client):
    """A figure's selects are its own control, not a stack of settings.

    The page's base rule stretches every `select` to full width, which is right
    for a form field and wrong here: three full-width selects in a card become
    three rows, and the axis pickers stop reading as one choice about one
    figure. So they go in a `.selectors` row, which sizes them to their
    contents. This asserts the row exists and that nothing escaped it, since a
    select added outside one would silently go back to a row of its own.
    """
    import re

    html = _page(client, _experiment())
    cards = re.findall(r'<section class="card" data-figure="([a-z_]+)"(.*?)</section>',
                       html, re.S)
    with_pickers = 0

    for key, body in cards:
        selects = [m.start() for m in re.finditer(r"<select", body)]
        if not selects:
            continue
        with_pickers += 1
        # Modifier classes are allowed beside it — a figure that needs its row
        # to hold its height still has one row holding its pickers.
        rows = [(m.start(), body.index("</p>", m.start()))
                for m in re.finditer(r'<p class="selectors[^"]*">', body)]
        assert rows, f"{key} has pickers but no row to hold them"
        for at in selects:
            assert any(start < at < end for start, end in rows), \
                f"{key} has a select outside its .selectors row"

    assert with_pickers >= 4, "the figures with pickers are all still on the page"


def test_each_reading_of_the_interactions_is_its_own_figure(client):
    """Five figures over one computation, not five views of one figure.

    They answer different questions and are wanted side by side — the heatmap is
    a grid you scan, the graph a shape you recognise, the coalitions a ranked
    list you read — and behind a single selector only one could ever be on the
    page at a time. Each now has its own visibility setting, and none of them
    has a view selector left.
    """
    keys = [f.key for f in FIGURES if f.key.startswith("interactions_")]
    html = _page(client, _experiment())

    assert keys == ["interactions_heatmap", "interactions_top_pairs",
                    "interactions_graph", "interactions_coalitions",
                    "interactions_orders"]
    for key in keys:
        assert FIGURES_BY_KEY[key].per_metric
        # Their only `views` are the three games, and nothing switches those
        # per figure — the page's one selector switches all of them together.
        assert list(FIGURES_BY_KEY[key].views) == list(HP_GAME_FIELDS)
        assert f'data-figure="{key}"' in html


def test_one_warning_reaches_every_reading_of_the_interactions(client):
    """It is the tunability game's caveat and all five read that game, so a
    warning that appeared on only whichever was showing would be a warning most
    readers never saw."""
    html = _page(client, _experiment())
    keys = [f.key for f in FIGURES if f.key.startswith("interactions_")]

    assert html.count('class="alert warning interactions-warning"') == len(keys)


def test_the_page_reads_in_the_declared_order(client):
    """Catalog order is page order, and the widths pair the figures up: two
    half-width figures share a row, a full-width one takes the row to itself."""
    html = _page(client, _experiment())
    # Bounded at the column beside it, or the trials table would count as the
    # grid's last row — it is in neither the grid nor after it, it is next to it.
    grid = html.split('class="grid-2x2"', 1)[1].split('class="figure-column"', 1)[0]
    drawn = [key for key in EXPECTED_KEYS if f'data-figure="{key}"' in grid]

    assert drawn == [
        "best_configuration", "hyperparameter_importance",
        "performance_over_time", "configuration_cube",
        "interactions_heatmap", "interactions_top_pairs",
        "interactions_graph", "interactions_coalitions",
        "interactions_orders", "trial_duration",
        "parallel_coordinates", "partial_dependence",
        "local_explanation", "local_effects",
    ]
    # Not in the grid at all: it spans the grid and the column beside it, so it
    # is rendered under both rather than inside either.
    assert 'data-figure="acquisition_slice"' not in grid
    assert 'class="slot-page' in html
    spans = {key for key in drawn
             if "wide" in grid[grid.rindex('<div class="slot', 0,
                                           grid.index(f'data-figure="{key}"')):
                                grid.index(f'data-figure="{key}"')]}
    # `acquisition_slice` is absent: it is page-width, so it never enters the
    # grid to span anything within it.
    assert spans == {"performance_over_time", "configuration_cube",
                     "partial_dependence",
                     "local_explanation", "local_effects",
                     "parallel_coordinates"}


def test_a_figures_title_and_its_pickers_share_a_line(client):
    """The pickers say what the title is currently showing, so they belong to
    the heading rather than sitting under it as a caption. One row per figure,
    whether or not it has any."""
    import re

    html = _page(client, _experiment())
    cards = re.findall(r'<section class="card" data-figure="([a-z_]+)"(.*?)</section>',
                       html, re.S)

    for key, body in cards:
        head = body.split('<div class="card-head">', 1)
        assert len(head) == 2, f"{key} has no head row"
        head = head[1].split("</div>\n    </div>", 1)[0]
        assert "subheader" in head, key
        if "<select" in body:
            assert "<select" in head, f"{key} keeps its pickers out of its heading"


def test_the_trials_table_is_a_column_beside_the_grid(client):
    """It is the figure you look things up in while reading a chart, so on a
    wide enough window it sits beside them rather than after them.

    Two panes in the markup, one layout in CSS: the grid, and the column. On a
    narrow window they stack, which puts the table exactly where its catalog
    order already had it — last, full width — so the layout it leaves is the
    layout it falls back to, and nothing here has to know which is showing.
    """
    html = _page(client, _experiment())
    layout = html.split('class="figure-layout"', 1)[1]
    grid = layout.split('class="grid-2x2"', 1)[1].split('class="figure-column"', 1)[0]
    column = layout.split('class="figure-column"', 1)[1]

    assert FIGURES_BY_KEY["trials"].in_side_column
    assert [f.key for f in FIGURES if f.in_side_column] == ["trials"]
    assert 'data-figure="trials"' in column
    assert 'data-figure="trials"' not in grid
    assert 'data-figure="performance_over_time"' in grid
    # and it is still last, so stacking is the layout it already had
    assert FIGURES[-1].key == "trials"


def test_a_square_figure_gets_two_rows_as_well_as_two_columns(client):
    """Height is declared separately from width because they answer different
    questions: how much room a figure needs beside it, and how much under it.

    The projection figure is the one that needs both. Every one of its three
    views is a scatter over a space with no privileged direction, and a scatter
    in a single row is a strip — the vertical axis gets a fifth of the room the
    horizontal one does and reports a fifth of what it has to say.
    """
    html = _page(client, _experiment())
    grid = html.split('class="grid-2x2"', 1)[1]
    at = grid.index('data-figure="configuration_cube"')
    slot = grid[grid.rindex('<div class="slot', 0, at):at]

    assert FIGURES_BY_KEY["configuration_cube"].height == "double"
    assert "wide" in slot and "tall" in slot
    assert [f.key for f in FIGURES if f.height == "double"] == [
        "configuration_cube", "acquisition_slice"]
    # and nothing else grew a second row by accident. Two now: the cube, whose
    # content is square, and the acquisition slice, whose three stacked panels
    # are genuinely tall — both declared, neither incidental.
    assert grid.count(" tall") == 2


def test_a_projection_opens_flat(client):
    """Three dimensions of a space with no privileged direction is a shape to be
    rotated before it says anything; two is the reading you can take at a
    glance. Declared rather than left to option order, like the axis pickers'
    None."""
    html = _page(client, _experiment())
    picker = html.split('id="cube-dimensions"', 1)[1].split("</select>", 1)[0]

    assert '<option value="2" selected>' in picker
    assert picker.count("selected") == 1


def test_the_page_reads_its_view_selectors_before_drawing(client):
    """A browser restores a select's value across a reload, and a handler bound
    to `change` never runs for a value the reader did not just set.

    Which is what the method picker did: it came back saying "PLS" while the
    figure drew axes and the axis pickers sat beside it, and only agreed once it
    had been moved away and back. Every view selector is read once at startup
    instead, so the control and the view cannot disagree — including the game,
    which is one selector driving six figures.
    """
    html = _page(client, _experiment())
    script = html.split("readViewSelectors", 1)[1].split("})();", 1)[0]

    assert "currentView.configuration_cube = cubeMethodSelect.value" in script
    assert "syncCubePickers()" in script
    assert "currentView.hyperparameter_importance" in script
    assert "currentView.performance_over_time" in script
    assert "gameFigures.forEach" in script
    # and it runs before anything is drawn, or it would be describing a page
    # that had already been built from the wrong views
    assert html.index("readViewSelectors") < html.index("redraw(key, null)")
