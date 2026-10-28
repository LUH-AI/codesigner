"""The page tints a failed trial's row, flags its sidebar score as a
placeholder, and serves its traceback as a file. The fixture builds failures by
editing a stored result.
"""

import json
from typing import Any

import plotly.graph_objects as go
import pytest
from django.urls import reverse

from core import io
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR

TRACEBACK = 'Traceback (most recent call last):\n  File "m.py"\nValueError: no'


def _drawn(fig: go.Figure | None) -> Any:
    assert fig is not None
    return fig


def _rebuilt_result(exp):
    """`_rebuilt_result` returns the result the page rebuilds."""
    built = adapter.rebuild_experiment(exp)
    assert built is not None
    return built["result"]


@pytest.fixture
def experiment():
    """The trial at index 3 crashed with a traceback, and the one at index 5
    timed out without one, as `evaluate_trial` records.
    """
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    assert exp.data.result is not None
    data = exp.data.result["data"]
    metrics = list(exp.data.metric_names)

    data[3]["status"] = STATUS_CRASHED
    data[3]["cost"] = 1.0
    data[3]["scores"] = {m: 0.0 for m in metrics}
    data[3]["additional_info"] = {"error": "ValueError: no", "traceback": TRACEBACK}

    data[5]["status"] = STATUS_TIMEOUT
    data[5]["cost"] = 1.0
    data[5]["scores"] = {m: 0.0 for m in metrics}
    data[5]["additional_info"] = {"error": "past its 600s deadline"}

    exp.data.save(update_fields=["result"])
    return exp


# the table

def test_the_table_marks_the_failed_rows(client, experiment):
    """Failed rows get a tint and keep their stored zeros, which keeps the score
    columns sortable."""
    html = client.get(reverse("ui:experiment_detail", args=[experiment.pk])).content.decode()
    rows = [r for r in html.split("<tr ") if "data-trial-idx" in r]

    failed = [i for i, r in enumerate(rows) if 'class="failed"' in r]
    assert failed == [3, 5]
    assert "ValueError: no" in rows[3], "the reason is in reach without leaving the table"


# the sidebar

def _panel(client, exp, idx):
    return client.get(
        reverse("ui:trial_panel", args=[exp.pk]) + f"?metric=accuracy&idx={idx}"
    ).content.decode()


def test_the_panel_says_the_score_is_not_a_result(client, experiment):
    """Without the notice, the 0.0 would read as a measurement."""
    body = _panel(client, experiment, 3)

    assert "placeholder" in body
    assert "ValueError: no" in body
    assert "metric-value failed" in body, "struck through, not hidden"


def test_the_panel_links_the_traceback_when_there_is_one(client, experiment):
    assert "trial-traceback" in _panel(client, experiment, 3)


def test_the_panel_offers_no_link_when_there_is_no_traceback(client, experiment):
    """A timeout has no traceback, and its reason alone explains it."""
    body = _panel(client, experiment, 5)

    assert "past its 600s deadline" in body
    assert "trial-traceback" not in body


def test_a_trial_that_succeeded_says_nothing_about_failure(client, experiment):
    body = _panel(client, experiment, 0)

    assert "placeholder" not in body
    assert "metric-value failed" not in body


# the traceback

def _traceback(client, exp, idx):
    return client.get(
        reverse("ui:trial_traceback", args=[exp.pk]) + f"?idx={idx}")


def test_the_traceback_is_served_as_plain_text(client, experiment):
    """`trial_traceback` responds inline, and a click opens the file in place."""
    response = _traceback(client, experiment, 3)

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/plain")
    assert response["Content-Disposition"].startswith("inline")
    body = response.content.decode()
    assert TRACEBACK in body
    assert "ValueError: no" in body, "the reason heads the file"


def test_a_trial_with_no_traceback_has_no_page(client, experiment):
    """The 404 covers success, timeout, and a traceback stripped by export."""
    assert _traceback(client, experiment, 5).status_code == 404
    assert _traceback(client, experiment, 0).status_code == 404


def test_an_index_outside_the_run_is_refused(client, experiment):
    assert _traceback(client, experiment, 9999).status_code == 400
    assert _traceback(client, experiment, "nope").status_code == 400


# the export

def _exported(client, exp, **answers):
    response = client.post(reverse("ui:experiment_export", args=[exp.pk]), answers)
    return json.loads(response.content.decode())["result"]["data"]


def test_the_export_asks_about_tracebacks(client, experiment):
    """A traceback, like the timestamps, describes the exporting machine instead
    of the search."""
    body = client.get(reverse("ui:experiment_export", args=[experiment.pk])).content.decode()

    assert 'name="tracebacks" value="keep"' in body
    assert "absolute paths" in body


def test_the_export_does_not_ask_when_there_is_nothing_to_send(client):
    clean = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    body = client.get(reverse("ui:experiment_export", args=[clean.pk])).content.decode()

    assert 'name="tracebacks"' not in body
    assert 'name="timestamps"' in body, "the other question is always asked"


def test_the_traceback_is_left_out_unless_it_is_asked_for(client, experiment):
    """The safe default strips the traceback."""
    data = _exported(client, experiment, timestamps="keep")

    assert "traceback" not in data[3]["additional_info"]
    assert data[3]["additional_info"]["error"] == "ValueError: no", (
        "the reason is the trial's own result and stays either way")


def test_the_traceback_travels_when_it_is_asked_for(client, experiment):
    data = _exported(client, experiment, timestamps="keep", tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK


def test_the_two_answers_are_independent(client, experiment):
    """Keeping the traceback must leave the timestamps out."""
    data = _exported(client, experiment, tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK
    assert "starttime" not in data[3]
    assert "endtime" not in data[3]


def test_a_failure_survives_a_round_trip(client, experiment):
    """A re-import must keep the status and reason, which belong to the trial's
    result."""
    from core.optimizers import RandomOptimizer

    body = client.post(reverse("ui:experiment_export", args=[experiment.pk]),
                       {"timestamps": "keep", "tracebacks": "keep"}).content
    reimported = adapter.experiment_from_snapshot(io.parse(body))
    assert reimported.data.result is not None
    result = RandomOptimizer().deserialize_result(reimported.data.result)

    assert [i for i, t in enumerate(result.trials) if t.failed] == [3, 5]
    assert result.trials[3].traceback == TRACEBACK
    assert result.trials[5].traceback == "", "it never had one"


# selection versus failure marks

def _script():
    from pathlib import Path

    return Path("ui/templates/ui/experiment_detail.html").read_text(encoding="utf-8")


def _recolor_branch():
    """`_recolor_branch` extracts `applySelection`'s repaint half."""
    source = _script()
    start = source.index("function applySelection(")
    end = source.index("const trialsPager", start)
    return source[start:end]


def test_the_selection_repaints_from_what_was_drawn_not_from_a_default():
    """The "recolor" repaint must restore the drawn colours and sizes, or
    selecting any trial flattens every failure mark.
    """
    branch = _recolor_branch()

    assert "_baseMarkers" in branch, "the repaint no longer restores what was drawn"
    assert "baseColor" in branch and "baseSize" in branch


def test_the_drawn_marker_styling_is_captured_before_anything_touches_it():
    """`draw` captures markers from the payload instead of the selection-tainted
    DOM."""
    source = _script()
    start = source.index("function draw(key, plot)")
    end = source.index("function clearPlot", start)

    assert "_baseMarkers" in source[start:end]
    assert "plot.data" in source[start:end], "captured from the payload, not the element"


def test_selecting_a_failure_keeps_its_cross_and_changes_its_colour():
    """A cross has no fill, so the selection colours a failure's outline."""
    branch = _recolor_branch()

    assert "marker.line.color" in branch
    assert "isFailure" in branch


# The projections mark selection with an outline because the fill shows the
# score, and a failure's cross is all outline.


def _ordinary_point_size():
    """`_ordinary_point_size` reads a successful cube point's size from the
    builder.
    """
    from ui.figures.plots import configuration_cube_plot

    from tests.ui.results.test_failed_trials import _result

    marker = _drawn(configuration_cube_plot(_result(failed_at=(2,)), "accuracy")).data[0].marker
    return marker.size[0]


def _outline_branch():
    """`_outline_branch` extracts the outline half of `applySelection`."""
    branch = _recolor_branch()
    start = branch.index('if (sel.style === "outline")')
    return branch[start:branch.index("Plotly.restyle(el, update", start)]


def test_the_projection_carries_failure_marks_at_all(client, experiment):
    """The server must send the failure marks on every view of the cube."""
    from ui.figures.plots import configuration_cube_plot

    result = _rebuilt_result(experiment)
    figure = _drawn(configuration_cube_plot(result, "accuracy"))

    symbols = list(figure.data[0].marker.symbol)
    assert [i for i, s in enumerate(symbols) if s != "circle"] == [3, 5]
    assert list(figure.data[0].marker.line.color)[3] == "#FF2B2B"


def test_the_outline_selection_keeps_a_failures_cross_visible():
    """An outline width of 0 erases an open cross, which hid failures while any
    point was selected.
    """
    branch = _outline_branch()

    assert "isFailure" in branch, "the ring no longer knows which points are crosses"
    assert "baseLineWidth" in branch, "a failure's own outline width is not restored"


def test_the_outline_selection_keeps_a_failures_colour():
    """Failures keep red; only the clicked point gets selection colour."""
    branch = _outline_branch()

    assert "baseLine[i]" in branch


def test_a_three_dimensional_view_uses_a_symbol_a_scene_can_draw():
    """A 3D scene draws no marker outline, so the outline-only `x-thin` becomes
    `x` there and loses its red.
    """
    branch = _recolor_branch()

    assert 'scene && sym === "x-thin" ? "x"' in branch
    assert '"marker.symbol"' in branch


def test_the_three_dimensional_cross_is_smaller_than_the_points_around_it():
    """Marker size reads as depth in a scene, and a cross looks wider than a
    disc of equal size.
    """
    import re

    from ui.figures.plots import FAILURE_SIZE

    source = _script()
    found = re.search(r"const FAILURE_SIZE_3D = (\d+);", source)
    assert found is not None
    size = int(found.group(1))

    assert size < _ordinary_point_size(), \
        "a failure should not be the largest thing on the plot"
    assert size < FAILURE_SIZE, "the flat mark is the one that has to carry itself"
    assert "Math.min(was, FAILURE_SIZE_3D)" in _recolor_branch()


def test_the_cube_rebuilds_its_marker_from_the_payload_not_the_element():
    """`applyCubeAxes` rebuilds the trace on every axis change, and a marker
    read off the element carries the previous view's edits. The rebuild starts
    from the payload each time instead.
    """
    cube = _cube_function()

    assert "baseMarker()" in cube
    assert "Object.assign({}, trace.marker)," not in cube, \
        "the marker is being rebuilt from the element again"


def test_the_drawn_marker_survives_a_change_of_dimensionality():
    """`applyCubeAxes` saves the drawn markers before `Plotly.newPlot` and
    restores them afterwards for `applySelection`.
    """
    source = _script()
    start = source.index("function applyCubeAxes(")
    # Whole function; `_cube_function` stops before the replot.
    whole = source[start:source.index('applySelection("configuration_cube"', start)]

    assert "const drawn = el._baseMarkers" in whole
    assert whole.index("Plotly.newPlot") < whole.index("el._baseMarkers = drawn")


def test_the_whole_marker_is_kept_not_only_what_the_selection_repaints():
    """The rebuild also needs the colour scale, colour bar and score array."""
    source = _script()
    start = source.index("function draw(key, plot)")
    capture = source[start:source.index("function clearPlot", start)]

    assert "JSON.parse(JSON.stringify(trace.marker" in capture


def test_the_flat_view_gets_its_open_cross_back_after_a_three_d_one():
    """The 3D symbol substitution reads the payload each time, because an edit
    on the element would stick after leaving the 3D view.
    """
    source = _script()
    start = source.index("function applyCubeAxes(")
    cube = source[start:source.index("Plotly.react(el, traces, layout", start)]

    assert '"x-thin"' not in cube, "applyCubeAxes is substituting symbols again"
    assert "baseSymbol.map" in _recolor_branch()


def _cube_function():
    source = _script()
    start = source.index("function applyCubeAxes(")
    return source[start:source.index("Plotly.react(el, traces, layout", start)]


def test_a_scene_is_not_handed_an_array_it_rejects():
    """`scatter3d.marker.line.width` is scalar-only, and Plotly drops the whole
    trace over an invalid attribute. The cube deletes `marker.line` in 3D, where
    a scene draws no outline anyway.
    """
    assert "delete newTrace.marker.line" in _cube_function()


def test_the_scene_draws_the_crosses_again_in_red():
    """A scene fills a failure with its pale score colour. A red overlay trace
    redraws the crosses and skips hover, and clicks still reach the real point.
    """
    cube = _cube_function()

    assert "selectionColors.failure" in cube
    assert 'hoverinfo: "skip"' in cube
    assert "traces.push" in cube


def test_the_extra_trace_goes_after_the_ones_the_selection_names():
    """The overlay goes after traces 0 and 1, which `layout.meta.selection`
    names."""
    cube = _cube_function()

    assert cube.index("const traces = [newTrace, highlight]") < cube.index(
        "selectionColors.failure")


@pytest.mark.django_db
def test_the_page_is_told_the_failure_colour(client, experiment):
    """The palette sends the failure colour, and the script hardcodes none."""
    import json

    from ui.figures import NEGATIVE_COLOR

    body = client.get(reverse("ui:experiment_detail",
                              args=[experiment.pk])).content.decode()
    start = body.index('id="selection-colors-data"')
    start = body.index(">", start) + 1
    colors = json.loads(body[start:body.index("</script>", start)])

    assert colors["failure"] == NEGATIVE_COLOR


def test_the_three_d_trace_the_page_builds_is_one_plotly_accepts(experiment):
    """The test applies the page's `scatter3d` transformation to the payload and
    validates the result against Plotly's schema.
    """
    import plotly.graph_objects as go

    from ui.figures.plots import configuration_cube_plot

    result = _rebuilt_result(experiment)
    marker = _drawn(configuration_cube_plot(result, "accuracy")).data[0].marker.to_plotly_json()
    assert isinstance(marker["line"]["width"], (list, tuple)), "no failures to carry"

    # Mirror of applyCubeAxes entering a scene.
    marker.pop("line")
    marker["symbol"] = [("x" if s == "x-thin" else s) for s in marker["symbol"]]

    points = list(range(len(result.trials)))
    go.Scatter3d(x=points, y=points, z=points, mode="markers", marker=marker)


def test_the_same_payload_is_rejected_without_that_transformation(experiment):
    """Rejection here keeps the test above honest."""
    import plotly.graph_objects as go

    from ui.figures.plots import configuration_cube_plot

    result = _rebuilt_result(experiment)
    marker = _drawn(configuration_cube_plot(result, "accuracy")).data[0].marker.to_plotly_json()
    points = list(range(len(result.trials)))

    with pytest.raises(ValueError):
        go.Scatter3d(x=points, y=points, z=points, mode="markers", marker=marker)
