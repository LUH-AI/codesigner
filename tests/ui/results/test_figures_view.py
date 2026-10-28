"""The detail page inlines the opening metric's plots and `ui:metric_figures`
serves the rest. `_figure_plots` builds both, and `_plots` reads either.
"""

import json

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _detail(client):
    """`_detail` renders the 30-trial fixture's html and experiment."""
    from experiments.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html, exp


def _detail_with_analytics(client):
    """`analytics.ihpo` comes from a real run and carries every analytics field
    that `test2.ihpo` predates.
    """
    from experiments.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    return html, exp


def _json_script(html, element_id):
    """`_json_script` parses one `json_script` payload."""
    start = html.index(f'id="{element_id}"')
    return json.loads(html[html.index(">", start) + 1: html.index("</script>", start)])


def _plots(client, exp, html, metric):
    """`_plots` fetches one metric's plots, inlined or from the switcher
    endpoint.
    """
    inline = _json_script(html, "metric-plots-data")
    if metric in inline:
        return inline[metric]
    resp = client.get(reverse("ui:metric_figures", args=[exp.pk]) + f"?metric={metric}")
    assert resp.status_code == 200, metric
    return resp.json()


def test_detail_includes_plotly_and_chart_containers(client):
    html, _ = _detail(client)
    assert "plotly.min.js" in html
    assert 'id="figure-performance_over_time"' in html
    assert 'id="figure-hyperparameter_importance"' in html


def test_detail_has_metric_switcher_with_all_metrics(client):
    html, exp = _detail(client)
    assert 'id="metric-select"' in html
    for m in exp.data.metric_names:
        assert f'value="{m}"' in html


def test_the_page_ships_only_the_metric_it_opens_on(client):
    """Inlining every metric took 72 `plot()` calls and ~300KB of JSON per load."""
    html, exp = _detail(client)
    inline = _json_script(html, "metric-plots-data")

    assert list(inline) == ["accuracy"], "the metric the selector opens on"
    assert len(exp.data.metric_names) > 1, "and there are others to fetch"


def test_every_metric_has_a_performance_figure_for_the_switcher(client):
    html, exp = _detail(client)
    for m in exp.data.metric_names:
        plots = _plots(client, exp, html, m)
        assert plots["performance_over_time"]["trial-score"]["data"]


def test_best_config_panel_present_per_metric(client):
    html, exp = _detail(client)
    for m in exp.data.metric_names:
        assert f'data-metric="{m}"' in html


def test_performance_over_time_offers_all_four_views(client):
    """Trial/time x score/error make four views."""
    html, exp = _detail(client)
    assert 'id="view-select-performance_over_time-x"' in html
    assert 'id="view-select-performance_over_time-y"' in html

    for m in exp.data.metric_names:
        views = _plots(client, exp, html, m)["performance_over_time"]
        assert set(views) == {"trial-score", "trial-error", "time-score", "time-error"}
        for view in views.values():
            assert view["data"]


def test_hyperparameter_importance_offers_three_games_and_three_renderings(client):
    """Three games x pie/bar/table give 9 view keys, and the table payload is
    None because the template renders the table."""
    html, exp = _detail(client)
    assert 'id="view-select-hyperparameter_importance-rendering"' in html

    games = ("tunability", "sensitivity", "mistunability")
    for m in exp.data.metric_names:
        views = _plots(client, exp, html, m)["hyperparameter_importance"]
        assert set(views) == {f"{g}-{r}" for g in games for r in ("pie", "bar", "table")}
        for game in games:
            assert views[f"{game}-table"] is None
            assert f'class="importance-table" data-metric="{m}" data-game="{game}"' in html
        # only tunability has data on this fixture
        assert views["tunability-pie"]["data"]
        assert views["tunability-bar"]["data"]


def test_table_view_suppresses_the_generic_empty_message(client):
    """`syncImportanceTable` hides the empty caption, which `draw()` would show
    for the table's None payload."""
    html, _ = _detail(client)
    assert "syncImportanceTable" in html
    assert "figure-hyperparameter_importance-empty" in html
    script = html[html.index("function syncImportanceTable"):]
    script = script[:script.index("function show(metric)")]
    assert "empty.hidden = true" in script


def test_figure_views_lists_only_multiview_figures(client):
    """The script reads `figure-views-data` to tell a keyed-by-view payload from
    a plain one."""
    html, _ = _detail(client)
    views = _json_script(html, "figure-views-data")

    assert set(views) == {
        "hyperparameter_importance", "performance_over_time",
        # The three games are the interaction figures' views, switched together
        # by the sidebar selector.
        "interactions_heatmap", "interactions_top_pairs", "interactions_graph",
        "interactions_coalitions", "interactions_orders",
        # Methods only; the axis assignment rides in customdata.
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
    """The one trace carries every hyperparameter in customdata for
    `applyCubeAxes`, because axis assignment is unbounded."""
    html, exp = _detail(client)
    assert 'id="cube-axis-x"' in html
    assert 'id="cube-axis-z"' in html

    for m in exp.data.metric_names:
        cube = _plots(client, exp, html, m)["configuration_cube"]["axes"]
        hp_names = cube["layout"]["meta"]["hp_names"]
        assert len(hp_names) >= 2
        trace = cube["data"][0]
        assert trace["customdata"], "the values the client plots from"
        assert all(len(row) == len(hp_names) for row in trace["customdata"])
        # Pickers start at None, so the server picks no pair.
        assert not trace["x"] and not trace["y"]
        assert len(trace["marker"]["color"]) == len(trace["customdata"])


def test_every_cube_axis_starts_at_none(client):
    """The count of filled pickers decides the figure's dimensionality, and a
    pre-filled one would add an unasked axis."""
    html, _exp = _detail(client)
    cube = html.split('data-figure="configuration_cube"', 1)[1].split("</section>", 1)[0]

    for axis in ("x", "y", "z"):
        picker = cube.split(f'id="cube-axis-{axis}"', 1)[1].split("</select>", 1)[0]
        assert '<option value="" selected>' in picker, axis
        assert picker.count("selected") == 1, axis


def test_parallel_coordinates_renders_one_dimension_per_hyperparameter_plus_score(client):
    """Axis order is a fixed tunability ranking per metric, so one plot per
    metric suffices."""
    html, exp = _detail(client)

    assert exp.data.result is not None
    hp_count = len(next(iter(exp.data.result["configs"].values())))

    for m in exp.data.metric_names:
        parcoords = _plots(client, exp, html, m)["parallel_coordinates"]
        labels = parcoords["layout"]["xaxis"]["ticktext"]
        assert len(labels) == hp_count + 1
        assert labels[-1] == m.capitalize()


def test_partial_dependence_ships_a_picker_but_no_precomputed_data(client):
    """Each payload keeps the figure key with a None value, and the browser
    fetches the data from the partial_dependence endpoint."""
    html, exp = _detail(client)
    assert 'id="pdp-hp-select"' in html
    assert exp.data.result is not None
    for h in next(iter(exp.data.result["configs"].values())):
        assert f'value="{h}"' in html

    for m in exp.data.metric_names:
        assert _plots(client, exp, html, m)["partial_dependence"] is None


INTERACTION_FIGURES = ["interactions_heatmap", "interactions_top_pairs",
                       "interactions_graph", "interactions_coalitions",
                       "interactions_orders"]


def test_the_interaction_readings_are_five_figures_not_five_views(client):
    """Each interaction reading is its own figure, served from the stored result
    with a payload per game.
    """
    html, exp = _detail_with_analytics(client)

    for m in exp.data.metric_names:
        plots = _plots(client, exp, html, m)
        for key in INTERACTION_FIGURES:
            assert f'data-figure="{key}"' in html, key
            assert set(plots[key]) == {"tunability", "sensitivity", "mistunability"}, key
            assert "data" in plots[key]["tunability"], key


def test_hyperparameter_interactions_render_real_order_2_values(client):
    """Real order-2 HyperSHAP values must survive storage and the rebuild into a
    drawn heatmap and bar."""
    html, exp = _detail_with_analytics(client)

    for m in exp.data.metric_names:
        plots = _plots(client, exp, html, m)
        heatmap = plots["interactions_heatmap"]["tunability"]["data"][0]
        assert heatmap["type"] == "heatmap"
        assert len(heatmap["z"]) == len(heatmap["x"]) == len(heatmap["y"])
        assert any(any(cell for cell in row) for row in heatmap["z"]), "not all zero"
        assert plots["interactions_top_pairs"]["tunability"]["data"][0]["x"], \
            "at least one pair to rank"


def test_the_by_order_figure_renders_from_the_moebius_decomposition(client):
    """Only `interactions_orders` reads the Möbius decomposition; the heatmap
    and top pairs read the FSII grid."""
    html, exp = _detail_with_analytics(client)

    for m in exp.data.metric_names:
        orders = _plots(client, exp, html, m)["interactions_orders"]["tunability"]
        assert orders["layout"]["barmode"] == "stack"
        assert [t["name"] for t in orders["data"]] == [
            "On its own", "In pairs", "In larger groups"]
        assert any(any(t["y"]) for t in orders["data"]), "not all zero"


def test_all_three_importance_games_render_real_numbers(client):
    """The check uses `analytics.ihpo` because `test2.ihpo` leaves sensitivity
    and mistunability empty."""
    html, exp = _detail_with_analytics(client)

    for m in exp.data.metric_names:
        views = _plots(client, exp, html, m)["hyperparameter_importance"]
        for game in ("tunability", "sensitivity", "mistunability"):
            assert views[f"{game}-pie"]["data"][0]["values"], game
            assert views[f"{game}-bar"]["data"][0]["y"], game
            assert views[f"{game}-table"] is None, "the template renders that one"


def test_each_game_has_its_own_numbers(client):
    """Identical games would mean a figure reads the wrong field."""
    html, exp = _detail_with_analytics(client)
    views = _plots(client, exp, html, "accuracy")["hyperparameter_importance"]

    values = {game: tuple(views[f"{game}-pie"]["data"][0]["values"])
              for game in ("tunability", "sensitivity", "mistunability")}

    assert len(set(values.values())) > 1, values


def test_the_configuration_figure_ships_all_three_of_its_readings(client):
    """A projection takes 0.2-5 ms, so all three readings ship with the page in
    `customdata` without axis assignment.
    """
    html, exp = _detail_with_analytics(client)

    for m in exp.data.metric_names:
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
        # Components have no units, so no log scaling applies.
        assert cube["pca"]["layout"]["meta"]["log_hps"] == []
