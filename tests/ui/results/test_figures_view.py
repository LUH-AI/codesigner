"""Result figures render on the detail page.

Figures used to be checked on the synchronous new_experiment response; with
create and run separated, they render on an experiment's detail page once it
has a result. We set up an experiment carrying a stored result (via the
adapter) and GET its detail page.

The page ships plot payloads for the metric it opens on only; every other
metric is fetched from `ui:metric_figures` the first time it is selected, so
the per-metric assertions here go through `_plots`, which reads whichever of
the two applies. Both come from the same `_figure_plots` in ui/views.py, which
is the point — the switcher must not get a different shape from the page.

DB access + isolated media come from tests/ui/conftest.py.
"""

import json

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _detail(client):
    """Create an experiment with the 30-trial fixture result; return (html, exp)."""
    from ui.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html, exp


def _detail_with_analytics(client):
    """The same, on the fixture that actually carries every analytics field.

    `test2.ihpo` predates sensitivity/mistunability/interactions, so it
    deserializes them as empty — fine for the shape assertions it backs, but it
    means nothing there checks that real HyperSHAP output reaches a rendered
    figure. This one is produced by a real run; see
    tests/fixtures/make_analytics_fixture.py.
    """
    from ui.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html, exp


def _json_script(html, element_id):
    """One `json_script` payload out of the rendered page."""
    start = html.index(f'id="{element_id}"')
    return json.loads(html[html.index(">", start) + 1: html.index("</script>", start)])


def _plots(client, exp, html, metric):
    """One metric's per-figure plot payloads, from wherever the page puts them.

    The opening metric is inlined; the rest come from the endpoint the metric
    switcher fetches. Reading both through one helper keeps every assertion
    below about the figures rather than about which of the two paths served
    them.
    """
    inline = _json_script(html, "metric-plots-data")
    if metric in inline:
        return inline[metric]
    resp = client.get(reverse("ui:metric_figures", args=[exp.pk]) + f"?metric={metric}")
    assert resp.status_code == 200, metric
    return resp.json()


def test_detail_includes_plotly_and_chart_containers(client):
    """The detail page loads Plotly and provides the figure mount points."""
    html, _ = _detail(client)
    assert "plotly.min.js" in html
    assert 'id="figure-performance_over_time"' in html
    assert 'id="figure-hyperparameter_importance"' in html


def test_detail_has_metric_switcher_with_all_metrics(client):
    """A metric selector offers every scored metric."""
    html, exp = _detail(client)
    assert 'id="metric-select"' in html
    for m in exp.metric_names:
        assert f'value="{m}"' in html


def test_the_page_ships_only_the_metric_it_opens_on(client):
    """The other metrics are fetched on selection instead. Building all of them
    was 72 `plot()` calls and ~300KB of JSON per load (a megabyte at 500
    trials) for payloads most visits never looked at."""
    html, exp = _detail(client)
    inline = _json_script(html, "metric-plots-data")

    assert list(inline) == ["accuracy"], "the metric the selector opens on"
    assert len(exp.metric_names) > 1, "and there are others to fetch"


def test_every_metric_has_a_performance_figure_for_the_switcher(client):
    """Whichever way it is served — one entry per declared view, the default
    (trial-score) among them."""
    html, exp = _detail(client)
    for m in exp.metric_names:
        plots = _plots(client, exp, html, m)
        assert plots["performance_over_time"]["trial-score"]["data"]


def test_best_config_panel_present_per_metric(client):
    """Each metric gets its own best-configuration panel to toggle between."""
    html, exp = _detail(client)
    for m in exp.metric_names:
        assert f'data-metric="{m}"' in html


def test_performance_over_time_offers_all_four_views(client):
    """Trial/time x score/error compose to four views, each carrying data."""
    html, exp = _detail(client)
    assert 'id="view-select-performance_over_time-x"' in html
    assert 'id="view-select-performance_over_time-y"' in html

    for m in exp.metric_names:
        views = _plots(client, exp, html, m)["performance_over_time"]
        assert set(views) == {"trial-score", "trial-error", "time-score", "time-error"}
        for view in views.values():
            assert view["data"]


def test_hyperparameter_importance_offers_three_games_and_three_renderings(client):
    """Tunability, sensitivity and mistunability x pie/bar/table — 9 flat view
    keys. pie/bar are plot payloads, table is None (the template renders it
    directly) but the table markup itself is present, one block per metric
    per game."""
    html, exp = _detail(client)
    assert 'id="view-select-hyperparameter_importance-rendering"' in html

    games = ("tunability", "sensitivity", "mistunability")
    for m in exp.metric_names:
        views = _plots(client, exp, html, m)["hyperparameter_importance"]
        assert set(views) == {f"{g}-{r}" for g in games for r in ("pie", "bar", "table")}
        for game in games:
            assert views[f"{game}-table"] is None
            assert f'class="importance-table" data-metric="{m}" data-game="{game}"' in html
        # tunability always has real data on this fixture; sensitivity and
        # mistunability's pie/bar payloads at least round-trip (possibly
        # empty on a fixture that predates them).
        assert views["tunability-pie"]["data"]
        assert views["tunability-bar"]["data"]


def test_table_view_suppresses_the_generic_empty_message(client):
    """The table view's payload is None by design (the template renders it
    directly, not from a plot) — the same signal draw() otherwise reads as
    "no data" and shows the empty-message caption for. A browser check caught
    this actually happening (the caption showing over a populated table); the
    JS-side fix isn't executed by this suite, so this pins the wiring the way
    test_scale_button.py already does for other client-side behavior."""
    html, _ = _detail(client)
    assert "syncImportanceTable" in html
    assert "figure-hyperparameter_importance-empty" in html
    script = html[html.index("function syncImportanceTable"):]
    script = script[:script.index("function show(metric)")]
    assert "empty.hidden = true" in script


def test_figure_views_lists_only_multiview_figures(client):
    """Single-view figures (trial_duration, the tables) never appear here —
    the script uses this to tell a plain payload from a keyed-by-view one
    without guessing from its shape."""
    html, _ = _detail(client)
    views = _json_script(html, "figure-views-data")

    assert set(views) == {
        "hyperparameter_importance", "performance_over_time",
        # The three games are the interaction figures' views: a game is a
        # different set of numbers to draw rather than a different way to draw
        # them, but "one precomputed payload per option" is exactly the plumbing
        # that needs. Nothing switches them individually — the sidebar's one
        # selector switches all of them together.
        "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
        "interactions_coalitions", "interactions_orders",
        # Its three methods: axes, and the two projections. Which hyperparameter
        # sits on which axis is not one of them — that is unbounded, and rides
        # in customdata instead.
        "configuration_cube",
    }
    assert views["hyperparameter_importance"] == [
        "tunability-pie", "tunability-bar", "tunability-table",
        "mistunability-pie", "mistunability-bar", "mistunability-table",
        "sensitivity-pie", "sensitivity-bar", "sensitivity-table",
    ]
    assert views["interactions_heatmap"] == [
        "tunability", "mistunability", "sensitivity"]
    assert views["configuration_cube"] == ["axes", "pca", "pls"]
    assert views["performance_over_time"] == [
        "trial-score", "trial-error", "time-score", "time-error"]


def test_configuration_cube_ships_every_hyperparameter_for_client_side_remap(client):
    """Which hyperparameters are on which axis is an unbounded per-experiment
    combination, not a fixed set the server precomputes one JSON per option for
    (contrast the *method*, which is three options and so is a view) — instead
    the one shipped trace carries every hyperparameter's values in customdata,
    plus hp_names (the column order) in layout.meta, for
    experiment_detail.html's applyCubeAxes to plot client-side."""
    html, exp = _detail(client)
    assert 'id="cube-axis-x"' in html
    assert 'id="cube-axis-z"' in html

    for m in exp.metric_names:
        cube = _plots(client, exp, html, m)["configuration_cube"]["axes"]
        hp_names = cube["layout"]["meta"]["hp_names"]
        assert len(hp_names) >= 2
        trace = cube["data"][0]
        assert trace["customdata"], "the values the client plots from"
        assert all(len(row) == len(hp_names) for row in trace["customdata"])
        # And no axis assignment: the pickers start at None, so the server
        # choosing a pair would decide the figure's question before it is asked.
        assert not trace["x"] and not trace["y"]
        assert len(trace["marker"]["color"]) == len(trace["customdata"])


def test_every_cube_axis_starts_at_none(client):
    """All three pickers, not just the third: how many are filled is what
    decides whether the figure draws nothing, a strip, a plane or a cube, so
    any of them pre-filled would be an axis nobody asked for."""
    html, _exp = _detail(client)
    cube = html.split('data-figure="configuration_cube"', 1)[1].split("</section>", 1)[0]

    for axis in ("x", "y", "z"):
        picker = cube.split(f'id="cube-axis-{axis}"', 1)[1].split("</select>", 1)[0]
        assert '<option value="" selected>' in picker, axis
        assert picker.count("selected") == 1, axis


def test_parallel_coordinates_renders_one_dimension_per_hyperparameter_plus_score(client):
    """No `views` entry — axis order is a full, fixed HyperSHAP-tunability
    ranking per metric, not a per-experiment unbounded combination like the
    cube's, so one precomputed plot per metric is enough."""
    html, exp = _detail(client)

    hp_count = len(next(iter(exp.result["configs"].values())))

    for m in exp.metric_names:
        parcoords = _plots(client, exp, html, m)["parallel_coordinates"]
        labels = parcoords["layout"]["xaxis"]["ticktext"]
        assert len(labels) == hp_count + 1
        assert labels[-1] == m.capitalize()


def test_partial_dependence_ships_a_picker_but_no_precomputed_data(client):
    """Which hyperparameter is showing is a picker choice, not one of a
    small precomputable set (contrast the three global games and the
    interactions heatmap/bar) — the figure key is present in every metric's
    payload (so a hidden-figure check like test_a_hidden_figure_ships_no_plot_data
    still has something to look for), but its value is always None; the
    real data comes from the partial_dependence endpoint instead, fetched
    client-side (see experiment_detail.html's refreshPartialDependence)."""
    html, exp = _detail(client)
    assert 'id="pdp-hp-select"' in html
    for h in next(iter(exp.result["configs"].values())):
        assert f'value="{h}"' in html

    for m in exp.metric_names:
        assert _plots(client, exp, html, m)["partial_dependence"] is None


INTERACTION_FIGURES = ["interactions_heatmap", "interactions_top_pairs",
                       "interactions_graph", "interactions_coalitions",
                       "interactions_orders"]


def test_the_interaction_readings_are_five_figures_not_five_views(client):
    """They answer different questions and are wanted side by side, so each is
    its own figure with its own visibility setting — behind one selector only
    one could ever be on the page at a time.

    All five come straight from the stored result: no lazy fetch like the
    importance figure's "local" game, and no view key to pick between.
    """
    html, exp = _detail_with_analytics(client)

    for m in exp.metric_names:
        plots = _plots(client, exp, html, m)
        for key in INTERACTION_FIGURES:
            assert f'data-figure="{key}"' in html, key
            assert set(plots[key]) == {"tunability", "sensitivity", "mistunability"}, key
            assert "data" in plots[key]["tunability"], key


def test_hyperparameter_interactions_render_real_order_2_values(client):
    """The whole path, end to end: a real HyperSHAP order-2 computation, through
    `_extract_pairwise`, through serialization into `Experiment.result`, through
    the rebuild, into a drawn heatmap and bar.

    Every link was covered separately before this — `test_plots.py` renders
    hand-written dicts, `test_hp_importance.py` checks the extraction — and
    nothing joined them, because the only fixture with a result predates the
    field."""
    html, exp = _detail_with_analytics(client)

    for m in exp.metric_names:
        plots = _plots(client, exp, html, m)
        heatmap = plots["interactions_heatmap"]["tunability"]["data"][0]
        assert heatmap["type"] == "heatmap"
        assert len(heatmap["z"]) == len(heatmap["x"]) == len(heatmap["y"])
        assert any(any(cell for cell in row) for row in heatmap["z"]), "not all zero"
        assert plots["interactions_top_pairs"]["tunability"]["data"][0]["x"], \
            "at least one pair to rank"


def test_the_by_order_figure_renders_from_the_moebius_decomposition(client):
    """End to end for the new data: a real Möbius transform, through storage and
    the rebuild, into a stacked bar. The heatmap and the top pairs read the
    order-2 FSII grid instead, so this is the reading that exercises it."""
    html, exp = _detail_with_analytics(client)

    for m in exp.metric_names:
        orders = _plots(client, exp, html, m)["interactions_orders"]["tunability"]
        assert orders["layout"]["barmode"] == "stack"
        assert [t["name"] for t in orders["data"]] == [
            "On its own", "In pairs", "In larger groups"]
        assert any(any(t["y"]) for t in orders["data"]), "not all zero"


def test_all_three_importance_games_render_real_numbers(client):
    """Sensitivity and mistunability get the same end-to-end check tunability
    already had. On `test2.ihpo` both fields are empty, so their pie/bar
    payloads round-trip as None and prove nothing about the games."""
    html, exp = _detail_with_analytics(client)

    for m in exp.metric_names:
        views = _plots(client, exp, html, m)["hyperparameter_importance"]
        for game in ("tunability", "sensitivity", "mistunability"):
            assert views[f"{game}-pie"]["data"][0]["values"], game
            assert views[f"{game}-bar"]["data"][0]["y"], game
            assert views[f"{game}-table"] is None, "the template renders that one"


def test_each_game_has_its_own_numbers(client):
    """Three games answering three different questions about the same history —
    if they agreed exactly, something would be reading the wrong field."""
    html, exp = _detail_with_analytics(client)
    views = _plots(client, exp, html, "accuracy")["hyperparameter_importance"]

    values = {game: tuple(views[f"{game}-pie"]["data"][0]["values"])
              for game in ("tunability", "sensitivity", "mistunability")}

    assert len(set(values.values())) > 1, values


def test_the_configuration_figure_ships_all_three_of_its_readings(client):
    """Axes, PCA and PLS, all with the page. A projection costs 0.2-5 ms from
    twenty trials to five thousand, so a Compute button would take longer to
    press than the computation takes — and switching between them is then a
    switch between payloads that are already here, like every other view.

    All three carry their numbers in `customdata` with no axis assignment, which
    is what lets one piece of client code draw all three and lets a projection
    switch between two and three dimensions without a round trip.
    """
    html, exp = _detail_with_analytics(client)

    for m in exp.metric_names:
        cube = _plots(client, exp, html, m)["configuration_cube"]
        assert set(cube) == {"axes", "pca", "pls"}
        for view, payload in cube.items():
            trace = payload["data"][0]
            assert trace["customdata"], view
            assert not trace["x"] and not trace["y"], view
            assert payload["layout"]["meta"]["selection"]["trials"]["0"] == \
                list(range(len(trace["customdata"]))), view

        components = cube["pca"]["layout"]["meta"]["components"]
        assert len(components) == len(cube["pca"]["data"][0]["customdata"][0])
        assert components[0].startswith("PC 1 ("), "with its share of the variance"
        assert cube["pls"]["layout"]["meta"]["components"][0] == "PLS 1"
        # A component is a combination of hyperparameters and has no units to be
        # logarithmic in; the scaling that mattered happened before the projection.
        assert cube["pca"]["layout"]["meta"]["log_hps"] == []
