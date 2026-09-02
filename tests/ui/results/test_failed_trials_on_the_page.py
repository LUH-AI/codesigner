"""What the page does with a trial that produced no measurement.

The figures mark it (see test_failed_trials.py); this is the rest of the
surfacing. Three things a reader needs and none of which a mark on a chart can
carry: which row of the table it was, that the big number in the sidebar is a
placeholder rather than a result, and why it failed — the last of which is a
traceback, which is a file to open rather than a panel to read.

Built by editing a stored result rather than by running a model that crashes: a
failure is a `status` and an `additional_info` on the trial's data entry, so
everything here is reachable without an optimizer. `scratch/fabricate_failures.py`
does the same thing to a copy of a real experiment, for looking at.
"""

import json

import pytest
from django.urls import reverse

from core import io
from core.optimizers.timing import STATUS_CRASHED, STATUS_TIMEOUT
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR

TRACEBACK = 'Traceback (most recent call last):\n  File "m.py"\nValueError: no'


@pytest.fixture
def experiment():
    """An experiment whose trial at index 3 crashed and index 5 timed out.

    The crash carries a traceback and the timeout does not, which is what
    `evaluate_trial` records: a deadline expiring says nothing about where the
    model was when it ran out.
    """
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    data = exp.result["data"]
    metrics = list(exp.metric_names)

    data[3]["status"] = STATUS_CRASHED
    data[3]["cost"] = 1.0
    data[3]["scores"] = {m: 0.0 for m in metrics}
    data[3]["additional_info"] = {"error": "ValueError: no", "traceback": TRACEBACK}

    data[5]["status"] = STATUS_TIMEOUT
    data[5]["cost"] = 1.0
    data[5]["scores"] = {m: 0.0 for m in metrics}
    data[5]["additional_info"] = {"error": "past its 600s deadline"}

    exp.save(update_fields=["result"])
    return exp


# ── the table ────────────────────────────────────────────────────────────────

def test_the_table_marks_the_failed_rows(client, experiment):
    """Tinted rather than crossed: a row is a fill. Its zeros are shown as
    stored — blanking them would leave the score columns unsortable and hide
    that the search was handed those numbers."""
    html = client.get(reverse("ui:experiment_detail", args=[experiment.pk])).content.decode()
    rows = [r for r in html.split("<tr ") if "data-trial-idx" in r]

    failed = [i for i, r in enumerate(rows) if 'class="failed"' in r]
    assert failed == [3, 5]
    assert "ValueError: no" in rows[3], "the reason is in reach without leaving the table"


# ── the sidebar ──────────────────────────────────────────────────────────────

def _panel(client, exp, idx):
    return client.get(
        reverse("ui:trial_panel", args=[exp.pk]) + f"?metric=accuracy&idx={idx}"
    ).content.decode()


def test_the_panel_says_the_score_is_not_a_result(client, experiment):
    """The number above it reads as a measurement of 0.0 otherwise, which is the
    one thing it is not."""
    body = _panel(client, experiment, 3)

    assert "placeholder" in body
    assert "ValueError: no" in body
    assert "metric-value failed" in body, "struck through, not hidden"


def test_the_panel_links_the_traceback_when_there_is_one(client, experiment):
    """A dozen lines of paths and frames is the wrong shape for a sidebar."""
    assert "trial-traceback" in _panel(client, experiment, 3)


def test_the_panel_offers_no_link_when_there_is_no_traceback(client, experiment):
    """The timeout. A link to a 404 is worse than no link, and the reason on its
    own is the whole story for a deadline that expired."""
    body = _panel(client, experiment, 5)

    assert "past its 600s deadline" in body
    assert "trial-traceback" not in body


def test_a_trial_that_succeeded_says_nothing_about_failure(client, experiment):
    """The notice is absent rather than empty, so the panel keeps its shape for
    the trials that are the normal case."""
    body = _panel(client, experiment, 0)

    assert "placeholder" not in body
    assert "metric-value failed" not in body


# ── the traceback itself ─────────────────────────────────────────────────────

def _traceback(client, exp, idx):
    return client.get(
        reverse("ui:trial_traceback", args=[exp.pk]) + f"?idx={idx}")


def test_the_traceback_is_served_as_plain_text(client, experiment):
    """Something to scroll, search and paste elsewhere — and inline, so a click
    opens it rather than downloading it."""
    response = _traceback(client, experiment, 3)

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/plain")
    assert response["Content-Disposition"].startswith("inline")
    body = response.content.decode()
    assert TRACEBACK in body
    assert "ValueError: no" in body, "the reason heads the file"


def test_a_trial_with_no_traceback_has_no_page(client, experiment):
    """Which covers three cases at once: it succeeded, it timed out, or its
    traceback was left out of an export."""
    assert _traceback(client, experiment, 5).status_code == 404
    assert _traceback(client, experiment, 0).status_code == 404


def test_an_index_outside_the_run_is_refused(client, experiment):
    assert _traceback(client, experiment, 9999).status_code == 400
    assert _traceback(client, experiment, "nope").status_code == 400


# ── and what leaves in an export ─────────────────────────────────────────────

def _exported(client, exp, **answers):
    response = client.post(reverse("ui:experiment_export", args=[exp.pk]), answers)
    return json.loads(response.content.decode())["result"]["data"]


def test_the_export_asks_about_tracebacks(client, experiment):
    """A second question beside the timestamps one, because a traceback is the
    same kind of thing: a fact about this machine rather than about the search."""
    body = client.get(reverse("ui:experiment_export", args=[experiment.pk])).content.decode()

    assert 'name="tracebacks" value="keep"' in body
    assert "absolute paths" in body


def test_the_export_does_not_ask_when_there_is_nothing_to_send(client):
    """Asking about something the file does not contain is a question with no
    answer."""
    clean = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    body = client.get(reverse("ui:experiment_export", args=[clean.pk])).content.decode()

    assert 'name="tracebacks"' not in body
    assert 'name="timestamps"' in body, "the other question is always asked"


def test_the_traceback_is_left_out_unless_it_is_asked_for(client, experiment):
    """Unticked is the safer answer, so the default strips it."""
    data = _exported(client, experiment, timestamps="keep")

    assert "traceback" not in data[3]["additional_info"]
    assert data[3]["additional_info"]["error"] == "ValueError: no", (
        "the reason is the trial's own result and stays either way")


def test_the_traceback_travels_when_it_is_asked_for(client, experiment):
    """So an .ihpo sent to someone can carry the crash they need to see."""
    data = _exported(client, experiment, timestamps="keep", tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK


def test_the_two_answers_are_independent(client, experiment):
    """Two questions, not one with two settings: keeping the traceback must not
    smuggle the timestamps out with it."""
    data = _exported(client, experiment, tracebacks="keep")

    assert data[3]["additional_info"]["traceback"] == TRACEBACK
    assert "starttime" not in data[3]
    assert "endtime" not in data[3]


def test_a_failure_survives_a_round_trip(client, experiment):
    """The status and the reason are the trial's result, so re-importing the file
    has to produce a run the page marks the same way."""
    from core.optimizers import RandomOptimizer

    body = client.post(reverse("ui:experiment_export", args=[experiment.pk]),
                       {"timestamps": "keep", "tracebacks": "keep"}).content
    reimported = adapter.experiment_from_snapshot(io.parse(body))
    result = RandomOptimizer().deserialize_result(reimported.result)

    assert [i for i, t in enumerate(result.trials) if t.failed] == [3, 5]
    assert result.trials[3].traceback == TRACEBACK
    assert result.trials[5].traceback == "", "it never had one"


# ── the selection must not paint over the failure marks ─────────────────────

def _script():
    from pathlib import Path

    return Path("ui/templates/ui/experiment_detail.html").read_text(encoding="utf-8")


def _recolor_branch():
    """The half of `applySelection` that repaints a trace's points."""
    source = _script()
    start = source.index("function applySelection(")
    end = source.index("const trialsPager", start)
    return source[start:end]


def test_the_selection_repaints_from_what_was_drawn_not_from_a_default():
    """Found by looking at the running app, which is the only way it could have
    been: selecting any trial anywhere on the page used to flatten every failure
    mark on the two figures whose selection style is "recolor". The duration bar
    lost its tint outright — repainted the ordinary blue — and the crosses on
    trial performance shrank from their own size back to an ordinary point's.

    The repaint has to put back what the server drew, so the guard is that it
    reads the drawn values rather than painting a constant over everything that
    is not selected.
    """
    branch = _recolor_branch()

    assert "_baseMarkers" in branch, "the repaint no longer restores what was drawn"
    assert "baseColor" in branch and "baseSize" in branch


def test_the_drawn_marker_styling_is_captured_before_anything_touches_it():
    """From the payload rather than the DOM, which by the time a second
    selection happens already carries the first one."""
    source = _script()
    start = source.index("function draw(key, plot)")
    end = source.index("function clearPlot", start)

    assert "_baseMarkers" in source[start:end]
    assert "plot.data" in source[start:end], "captured from the payload, not the element"


def test_selecting_a_failure_keeps_its_cross_and_changes_its_colour():
    """`_failure_marks` says the shape carries the failure and the colour
    carries the selection. The client has to agree: a cross has no fill, so on a
    failure it is the outline that has to turn."""
    branch = _recolor_branch()

    assert "marker.line.color" in branch
    assert "isFailure" in branch



# ── …including on the figures whose colour already means something ──────────
#
# The two projections (the cube's axes view and its PCA/PLS views) style
# selection as an *outline* rather than a recolour, because the fill is the
# score. That branch painted a constant over every point's outline — and a
# failure's glyph is an open cross, so the outline *is* the mark.


def _ordinary_point_size():
    """What the cube draws a trial that succeeded at.

    Read off the builder rather than repeated here, so the size the 3D failure
    mark has to sit below cannot drift away from the size it is sitting among.
    """
    from ui.figures.plots import configuration_cube_plot

    from tests.ui.results.test_failed_trials import _result

    marker = configuration_cube_plot(_result(failed_at=(2,)), "accuracy").data[0].marker
    return marker.size[0]


def _outline_branch():
    """The half of `applySelection` that rings a trace's selected point."""
    branch = _recolor_branch()
    start = branch.index('if (sel.style === "outline")')
    return branch[start:branch.index("Plotly.restyle(el, update", start)]


def test_the_projection_carries_failure_marks_at_all(client, experiment):
    """Every view of it. The server has to send them before anything can keep
    them, and this is the figure the marks were missing from."""
    from ui.figures.plots import configuration_cube_plot
    from ui.views import _rebuild_experiment

    result = _rebuild_experiment(experiment)["result"]
    figure = configuration_cube_plot(result, "accuracy")

    symbols = list(figure.data[0].marker.symbol)
    assert [i for i, s in enumerate(symbols) if s != "circle"] == [3, 5]
    assert list(figure.data[0].marker.line.color)[3] == "#FF2B2B"


def test_the_outline_selection_keeps_a_failures_cross_visible():
    """The bug: an outline width of 0 erases an open cross entirely.

    Every point that was not the selected one had its outline zeroed, so on the
    hyperparameter space projection a failed trial looked exactly like a trial
    that scored badly — and something is always selected, so it was never
    visible at all.
    """
    branch = _outline_branch()

    assert "isFailure" in branch, "the ring no longer knows which points are crosses"
    assert "baseLineWidth" in branch, "a failure's own outline width is not restored"


def test_the_outline_selection_keeps_a_failures_colour():
    """Red says "not measured"; the selection colour is reserved for the one
    point that was clicked."""
    branch = _outline_branch()

    assert "baseLine[i]" in branch


def test_a_three_dimensional_view_uses_a_symbol_a_scene_can_draw():
    """`x-thin` is outline-only, and a 3D scene draws no marker outline.

    So the shape has to change with the dimensionality, or a failure in the 3D
    view is drawn as nothing at all. The red cannot come with it — the same
    limitation the selection ring already has in a scene.
    """
    branch = _recolor_branch()

    assert 'scene && sym === "x-thin" ? "x"' in branch
    assert '"marker.symbol"' in branch


def test_the_three_dimensional_cross_is_smaller_than_the_points_around_it():
    """Below the size the ordinary points are drawn at, not above it.

    A filled glyph covers its whole extent where an open one does not, a cross
    reads wider than a disc of the same nominal size, and in a scene size is a
    depth cue — an enlarged marker reads as one nearer the camera. The red is
    what carries the signal in 3D, so the shape does not have to.
    """
    import re

    from ui.figures.plots import FAILURE_SIZE

    source = _script()
    size = int(re.search(r"const FAILURE_SIZE_3D = (\d+);", source).group(1))

    assert size < _ordinary_point_size(), \
        "a failure should not be the largest thing on the plot"
    assert size < FAILURE_SIZE, "the flat mark is the one that has to carry itself"
    assert "Math.min(was, FAILURE_SIZE_3D)" in _recolor_branch()


def test_the_cube_rebuilds_its_marker_from_the_payload_not_the_element():
    """The root cause of two separate disappearing-cross bugs, pinned once.

    `applyCubeAxes` rebuilds the trace on every axis change, and whatever it
    builds becomes `el.data[0]` — so anything it reads off the element is
    whatever the *previous* view needed, not what the server drew. Adjusting a
    marker in place therefore feeds forward and never recovers:

    - substituting the 3D symbol left the open cross filled on the way back;
    - dropping `marker.line` for 3D (a scene rejects an array width) left the
      2D cross with no outline at all, which is the whole mark.

    Neither is fixed by undoing the specific adjustment. The fix is that the
    rebuild starts from the payload every time, which makes both self-correcting
    and any future one too.
    """
    cube = _cube_function()

    assert "baseMarker()" in cube
    assert "Object.assign({}, trace.marker)," not in cube, \
        "the marker is being rebuilt from the element again"


def test_the_drawn_marker_survives_a_change_of_dimensionality():
    """2D and 3D are different Plotly subplot types, so switching between them
    is a `newPlot` rather than a `react` — and `applySelection` runs afterwards
    and needs what the server drew. Carried across rather than assumed to
    survive: without it no point is a failure any more, and every cross is
    repainted as an ordinary ring.
    """
    source = _script()
    start = source.index("function applyCubeAxes(")
    # To the end of the function, which `_cube_function` stops short of: the
    # replot is the last thing it does.
    whole = source[start:source.index('applySelection("configuration_cube"', start)]

    assert "const drawn = el._baseMarkers" in whole
    assert whole.index("Plotly.newPlot") < whole.index("el._baseMarkers = drawn")


def test_the_whole_marker_is_kept_not_only_what_the_selection_repaints():
    """The rebuild needs the colour scale, the colour bar and the score array
    too, not just the four keys the repaint touches."""
    source = _script()
    start = source.index("function draw(key, plot)")
    capture = source[start:source.index("function clearPlot", start)]

    assert "JSON.parse(JSON.stringify(trace.marker" in capture


def test_the_flat_view_gets_its_open_cross_back_after_a_three_d_one():
    """The bug this caused: `applyCubeAxes` rebuilds the trace from whatever is
    on the element, so a symbol substituted in place while 3D was showing came
    back as the base on the next call and the open cross never returned.

    The guard is that the substitution reads the *payload* every time rather
    than the element, which is self-correcting in both directions.
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
    """`scatter3d.marker.line.width` is scalar-only, and `_failure_marks` makes
    it an array. Plotly takes the whole trace down over an invalid attribute, so
    a run with any failure in it drew *nothing at all* in three dimensions —
    not the crosses, not the other points either.

    Dropped rather than flattened to a scalar: a scene draws no marker outline,
    so `marker.line` carries nothing there in the first place.
    """
    assert "delete newTrace.marker.line" in _cube_function()


def test_the_scene_draws_the_crosses_again_in_red():
    """With no outline to carry it, a failure in a scene is filled with its own
    score — and a failure scores at the pale end of the ramp, so the crosses
    come out very nearly white.

    Redrawn on top as scenery, the same pattern the incumbent markers and the
    uncertainty field use: skipping hover takes it out of hit-testing so a click
    still reaches the real point underneath.
    """
    cube = _cube_function()

    assert "selectionColors.failure" in cube
    assert 'hoverinfo: "skip"' in cube
    assert "traces.push" in cube


def test_the_extra_trace_goes_after_the_ones_the_selection_names():
    """`layout.meta.selection` names trace 0 as the trials and trace 1 as the
    highlight, so anything inserted in front of them renumbers both."""
    cube = _cube_function()

    assert cube.index("const traces = [newTrace, highlight]") < cube.index(
        "selectionColors.failure")


@pytest.mark.django_db
def test_the_page_is_told_the_failure_colour(client, experiment):
    """Sent from the palette rather than repeated in the script, so the page
    restates no colour it does not own."""
    import json

    from ui.figures import NEGATIVE_COLOR

    body = client.get(reverse("ui:experiment_detail",
                              args=[experiment.pk])).content.decode()
    start = body.index('id="selection-colors-data"')
    start = body.index(">", start) + 1
    colors = json.loads(body[start:body.index("</script>", start)])

    assert colors["failure"] == NEGATIVE_COLOR


def test_the_three_d_trace_the_page_builds_is_one_plotly_accepts(experiment):
    """The check that would have caught it, done against Plotly's own schema.

    The client rebuilds the cube's trace as `scatter3d` from the server payload,
    and a scene's marker validates differently from a plane's — `line.width`
    array-valued is the one that bit, and an invalid attribute costs the whole
    trace rather than the attribute. So the payload is put through the same
    transformation the page performs and handed to the validator.
    """
    import plotly.graph_objects as go

    from ui.figures.plots import configuration_cube_plot
    from ui.views import _rebuild_experiment

    result = _rebuild_experiment(experiment)["result"]
    marker = configuration_cube_plot(result, "accuracy").data[0].marker.to_plotly_json()
    assert isinstance(marker["line"]["width"], (list, tuple)), "no failures to carry"

    # What applyCubeAxes does on the way into a scene.
    marker.pop("line")
    marker["symbol"] = [("x" if s == "x-thin" else s) for s in marker["symbol"]]

    points = list(range(len(result.trials)))
    go.Scatter3d(x=points, y=points, z=points, mode="markers", marker=marker)


def test_the_same_payload_is_rejected_without_that_transformation(experiment):
    """Otherwise the test above proves only that some dict validates."""
    import plotly.graph_objects as go

    from ui.figures.plots import configuration_cube_plot
    from ui.views import _rebuild_experiment

    result = _rebuild_experiment(experiment)["result"]
    marker = configuration_cube_plot(result, "accuracy").data[0].marker.to_plotly_json()
    points = list(range(len(result.trials)))

    with pytest.raises(ValueError):
        go.Scatter3d(x=points, y=points, z=points, mode="markers", marker=marker)
