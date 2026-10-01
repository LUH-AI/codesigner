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

import re

from django.urls import reverse

from ui.figures import (
    ACQUISITION, FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HYPERSHAP, LONG, MISC,
    OVERVIEW, SHAPES, SQUAT, TABLE, TABS, Figure,
)
from ui.models import Experiment, GlobalSettings

EXPECTED_KEYS = [
    # Rendered into the sidebar rather than the grid (Figure.in_sidebar), but
    # still a figure with a visibility setting like any other.
    "selected_configuration",
    "best_configuration",
    "hyperparameter_importance",
    "performance_over_time",
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
    # The acquisition function and the prior that weights it: one script and
    # one request, two figures.
    "acquisition_slice",
    "prior",
    # On the acquisition tab: where the search has been, beside where it would
    # go next.
    "configuration_cube",
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
    assert (Whatever.home_tab, Whatever.shape) == (MISC, SQUAT)
    assert Whatever.per_metric is False
    assert Whatever.absolute_scale is None
    assert Whatever.plot(result=None) is None


def test_each_figure_declares_a_tab_and_a_shape():
    """Every figure has a home tab and one of the four shapes, and opens at a
    width its shape allows — one, two or four columns, never three.

    The tabs gather what is read together: the HyperSHAP explanations, the
    acquisition function and its prior, and the rest of the run."""
    for figure in FIGURES:
        if figure.in_sidebar:
            continue
        assert figure.home_tab in TABS and figure.home_tab != OVERVIEW, figure.key
        assert figure.shape in SHAPES, figure.key
        assert 3 not in figure.columns(), "one, two or the whole row"

    by_tab = {tab: [f.key for f in FIGURES if f.home_tab == tab and not f.in_sidebar]
              for tab in TABS}
    assert by_tab[HYPERSHAP] == [
        "hyperparameter_importance", "interactions_heatmap", "interactions_top_pairs",
        "interactions_graph", "interactions_coalitions", "interactions_orders",
        "local_explanation", "local_effects"]
    assert by_tab[ACQUISITION] == ["acquisition_slice", "prior", "configuration_cube"]
    assert by_tab[OVERVIEW] == []
    # Read along an axis, so wider than tall and never narrower than two.
    assert [f.key for f in FIGURES if f.shape == LONG] == [
        "performance_over_time", "parallel_coordinates", "partial_dependence",
        "local_effects", "acquisition_slice", "prior"]
    assert FIGURES_BY_KEY["trials"].shape == TABLE
    assert SHAPES[TABLE] == (2,), "the trials table is always two columns"


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


def _panel(html, tab):
    """One tab's panel of the page."""
    return html.split(f'id="fig-panel-{tab}"', 1)[1].split('class="fig-panel"', 1)[0]


def test_the_selected_configuration_panel_is_not_in_a_tab(client):
    """It answers a click made anywhere on the page, so it lives in the sidebar
    (Figure.in_sidebar) where it stays readable while the figures are being
    clicked. It is still a figure — still in the catalog, still with a
    visibility setting — only somewhere else."""
    html = _page(client, _experiment())
    panels = html.split('class="fig-tabs"', 1)[1]
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]

    assert FIGURES_BY_KEY["selected_configuration"].in_sidebar
    assert [f.key for f in FIGURES if f.in_sidebar] == ["selected_configuration"]
    assert 'data-figure="selected_configuration"' in sidebar
    assert 'data-key="selected_configuration"' not in panels
    assert 'data-figure="best_configuration"' in panels, "the other panel stays"


def test_there_is_a_tab_bar_under_the_run_summary(client):
    html = _page(client, _experiment())

    bar = html.split('class="fig-tabs"', 1)[1].split("</div>", 1)[0]

    assert re.findall(r'data-tab="([a-z]+)"', bar) == list(TABS)
    assert html.index("Seed") < html.index('class="fig-tabs"'), "under the heading"


def test_each_figure_is_drawn_once_on_its_home_tab(client):
    """Rendered into one tab only, so its `#figure-<key>` id is unique and the
    page script draws into it without knowing about tabs."""
    html = _page(client, _experiment())

    for figure in FIGURES:
        if figure.in_sidebar:
            continue
        holding = [tab for tab in TABS
                   if f'data-figure="{figure.key}"' in _panel(html, tab)]
        assert holding == [figure.home_tab], figure.key
        assert html.count(f'id="{figure.dom_id}"') <= 1, figure.key


def test_overview_holds_empty_slots_for_what_is_pinned(client):
    """Filled by layout.js, which moves each figure into the slot on whichever
    tab is showing."""
    html = _page(client, _experiment())
    overview = _panel(html, OVERVIEW)

    from ui.layout import default_layout

    assert re.findall(r'data-key="([a-z_]+)"', overview) == [
        s["key"] for s in default_layout()["tabs"][OVERVIEW]]
    assert "data-figure=" not in overview


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


def test_each_tab_reads_in_the_declared_order(client):
    """The site's default layout file is each tab's order and widths until a
    reader rearranges it — see config/figure_layout.toml."""
    from ui.layout import default_layout

    html = _page(client, _experiment())

    for tab, slots in default_layout()["tabs"].items():
        panel = _panel(html, tab)
        found = re.findall(r'data-key="([a-z_]+)"\s+data-w="(\d)"', panel)
        assert [(k, int(w)) for k, w in found] == [(s["key"], s["w"]) for s in slots], tab


def test_a_slot_says_its_shape(client):
    """layout.js turns shape and width into rows and columns."""
    html = _page(client, _experiment())

    found = dict((key, shape) for shape, key in
                 re.findall(r'class="fig-slot shape-(\w+)" data-key="([a-z_]+)"', html))
    assert found["configuration_cube"] == SQUAT
    assert found["parallel_coordinates"] == LONG
    assert found["acquisition_slice"] == LONG
    assert found["prior"] == LONG
    assert found["trials"] == TABLE


def test_a_tab_whose_figures_are_all_off_says_so(client):
    _hide("acquisition_slice", "prior", "configuration_cube")
    html = _page(client, _experiment())

    assert "No figures on this tab" in _panel(html, ACQUISITION)


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
