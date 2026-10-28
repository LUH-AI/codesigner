"""Each catalog figure has a visibility setting, on by default, and hiding one
must leave the rest of the shared script working.
"""

import re

from django.urls import reverse

from ui.figures import (
    ACQUISITION, FIGURES, FIGURES_BY_KEY, HP_GAME_FIELDS, HYPERSHAP, LONG, MISC,
    OVERVIEW, SHAPES, SQUAT, TABLE, TABS, Figure,
)
from experiments.models import Experiment, GlobalSettings

EXPECTED_KEYS = [
    # Sidebar figure (`Figure.in_sidebar`) with its own visibility setting.
    "selected_configuration",
    "best_configuration",
    "hyperparameter_importance",
    "performance_over_time",
    "interactions_heatmap",
    "interactions_top_pairs",
    "interactions_graph",
    "interactions_coalitions",
    "interactions_orders",
    "trial_duration",
    # Forecast pages only (`Figure.forecast_only`).
    "forecast",
    "parallel_coordinates",
    "partial_dependence",
    # `local_explanation` is a separate figure because it explains one trial and
    # has no interactions.
    "local_explanation",
    "local_effects",
    "acquisition_slice",
    "prior",
    "configuration_cube",
    "trials",
]


def _experiment():
    """One finished trial gives every figure data."""
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
    """Settings keys and both pages read the catalog."""
    assert [c.key for c in FIGURES] == EXPECTED_KEYS


def test_every_figure_has_a_label_and_a_template():
    for figure in FIGURES:
        assert str(figure.label)
        assert figure.template == f"ui/figures/{figure.key}.html"
        assert figure.setting_key == f"show_{figure.key}"


def test_declaring_a_subclass_derives_everything_from_its_key():
    """`Figure` derives template, setting key and DOM id from `key`."""
    class Whatever(Figure):
        key = "whatever"
        label = "Whatever"

    assert Whatever.template == "ui/figures/whatever.html"
    assert Whatever.setting_key == "show_whatever"
    assert Whatever.dom_id == "figure-whatever"
    # unset traits take the common case
    assert (Whatever.home_tab, Whatever.shape) == (MISC, SQUAT)
    assert Whatever.per_metric is False
    assert Whatever.absolute_scale is None
    assert Whatever.plot(result=None) is None


def test_each_figure_declares_a_tab_and_a_shape():
    """Each figure has a home tab and one of the four shapes, opening one, two
    or four columns wide."""
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
    # Axis-read figures: wider than tall, at least two columns.
    assert [f.key for f in FIGURES if f.shape == LONG] == [
        "performance_over_time", "forecast", "parallel_coordinates", "partial_dependence",
        "local_effects", "acquisition_slice", "prior"]
    assert FIGURES_BY_KEY["trials"].shape == TABLE
    assert SHAPES[TABLE] == (2,), "the trials table is always two columns"


def test_only_metric_dependent_figures_are_marked_per_metric():
    """Per-metric figures rebuild on a metric change; trial duration and the
    tables stay the same."""
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
    """A figure gets the scale button by declaring its absolute range, one per
    view for `performance_over_time`.
    """
    scale = FIGURES_BY_KEY["performance_over_time"].absolute_scale
    assert scale is not None
    assert scale["trial-score"] == {"yaxis.range": [0, 1], "yaxis.autorange": False}
    assert scale["time-score"] == {"yaxis.range": [0, 1], "yaxis.autorange": False}
    # log y-axis, so the range is in log10 units
    assert scale["trial-error"] == {"yaxis.range": [-3, 0], "yaxis.autorange": False}
    assert scale["time-error"] == {"yaxis.range": [-3, 0], "yaxis.autorange": False}
    assert FIGURES_BY_KEY["trials"].absolute_scale is None


def _applies(figure):
    """`_applies` excludes forecast-only figures, since `_experiment()` has no
    forecast."""
    return not figure.forecast_only


def _applying_slots(slots):
    return [s for s in slots if _applies(FIGURES_BY_KEY[s["key"]])]


def test_all_figures_show_by_default(client):
    html = _page(client, _experiment())
    for figure in FIGURES:
        assert (f'data-figure="{figure.key}"' in html) == _applies(figure), figure.key


def test_unchecking_a_figure_removes_it_from_the_page(client):
    _hide("hyperparameter_importance")
    html = _page(client, _experiment())

    assert 'data-figure="hyperparameter_importance"' not in html
    # others stay
    assert 'data-figure="performance_over_time"' in html
    assert 'data-figure="trials"' in html


def test_hiding_the_performance_figure_keeps_the_page_working(client):
    """The click-to-select handler finds the performance figure through an id
    lookup."""
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
    """A hidden figure is never computed and leaves no entry in the plot JSON."""
    _hide("trial_duration", "performance_over_time")
    html = _page(client, _experiment())
    static = html.split('id="static-plots-data"', 1)[1].split("</script>", 1)[0]
    per_metric = html.split('id="metric-plots-data"', 1)[1].split("</script>", 1)[0]

    assert "trial_duration" not in static
    assert "performance_over_time" not in per_metric
    # visible figures unaffected
    assert "hyperparameter_importance" in per_metric


def _panel(html, tab):
    """`_panel` extracts one tab's panel."""
    return html.split(f'id="fig-panel-{tab}"', 1)[1].split('class="fig-panel"', 1)[0]


def test_the_selected_configuration_panel_is_not_in_a_tab(client):
    """`selected_configuration` sits in the sidebar to stay readable during
    clicks and stays in the catalog."""
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
    """A unique `#figure-<key>` id lets the script draw without knowing about
    tabs."""
    html = _page(client, _experiment())

    for figure in FIGURES:
        if figure.in_sidebar or not _applies(figure):
            continue
        holding = [tab for tab in TABS
                   if f'data-figure="{figure.key}"' in _panel(html, tab)]
        assert holding == [figure.home_tab], figure.key
        assert html.count(f'id="{figure.dom_id}"') <= 1, figure.key


def test_overview_holds_empty_slots_for_what_is_pinned(client):
    """`layout.js` fills the slots on the visible tab."""
    html = _page(client, _experiment())
    overview = _panel(html, OVERVIEW)

    from ui.layout import default_layout

    assert re.findall(r'data-key="([a-z_]+)"', overview) == [
        s["key"] for s in _applying_slots(default_layout()["tabs"][OVERVIEW])]
    assert "data-figure=" not in overview


def test_a_figures_pickers_sit_in_one_row_above_it(client):
    """Each figure's pickers sit in a `.selectors` row sized to their content,
    because the base rule stretches every `select` to full width.
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
        # Modifier classes on the row are allowed.
        rows = [(m.start(), body.index("</p>", m.start()))
                for m in re.finditer(r'<p class="selectors[^"]*">', body)]
        assert rows, f"{key} has pickers but no row to hold them"
        for at in selects:
            assert any(start < at < end for start, end in rows), \
                f"{key} has a select outside its .selectors row"

    assert with_pickers >= 4, "the figures with pickers are all still on the page"


def test_each_reading_of_the_interactions_is_its_own_figure(client):
    """Each interaction reading is its own figure with its own setting, because
    readers want them side by side.
    """
    keys = [f.key for f in FIGURES if f.key.startswith("interactions_")]
    html = _page(client, _experiment())

    assert keys == ["interactions_heatmap", "interactions_top_pairs",
                    "interactions_graph", "interactions_coalitions",
                    "interactions_orders"]
    for key in keys:
        assert FIGURES_BY_KEY[key].per_metric
        # Each figure's only views are the three games, switched together by the
        # page's one selector.
        assert list(FIGURES_BY_KEY[key].views) == list(HP_GAME_FIELDS)
        assert f'data-figure="{key}"' in html


def test_one_warning_reaches_every_reading_of_the_interactions(client):
    """The interactions warning appears on all five figures, so a reader of any
    one sees it."""
    html = _page(client, _experiment())
    keys = [f.key for f in FIGURES if f.key.startswith("interactions_")]

    assert html.count('class="alert warning interactions-warning"') == len(keys)


def test_each_tab_reads_in_the_declared_order(client):
    """`ui/figure_layout.toml` sets each tab's order and widths until a reader
    rearranges them."""
    from ui.layout import default_layout

    html = _page(client, _experiment())

    for tab, slots in default_layout()["tabs"].items():
        panel = _panel(html, tab)
        found = re.findall(r'data-key="([a-z_]+)"\s+data-w="(\d)"', panel)
        assert [(k, int(w)) for k, w in found] == [
            (s["key"], s["w"]) for s in _applying_slots(slots)], tab


def test_a_slot_says_its_shape(client):
    """Shapes and widths feed `layout.js`."""
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
    """The pickers describe what the title shows, so they share its line in the
    card head."""
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
    """A projection opens in 2D, readable at a glance, and the markup declares
    that default."""
    html = _page(client, _experiment())
    picker = html.split('id="cube-dimensions"', 1)[1].split("</select>", 1)[0]

    assert '<option value="2" selected>' in picker
    assert picker.count("selected") == 1


def test_the_page_reads_its_view_selectors_before_drawing(client):
    """Browsers restore a select's value on reload without firing `change`, so
    `readViewSelectors` reads every view selector once at startup.
    """
    html = _page(client, _experiment())
    script = html.split("readViewSelectors", 1)[1].split("})();", 1)[0]

    assert "currentView.configuration_cube = cubeMethodSelect.value" in script
    assert "syncCubePickers()" in script
    assert "currentView.hyperparameter_importance" in script
    assert "currentView.performance_over_time" in script
    assert "gameFigures.forEach" in script
    # `readViewSelectors` must run before the first draw.
    assert html.index("readViewSelectors") < html.index("redraw(key, null)")
