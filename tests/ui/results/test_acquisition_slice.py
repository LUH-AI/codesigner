"""The acquisition-and-priors figure, end to end through its endpoint.

This figure is unusual twice over: it ships no server-rendered plot (like
partial dependence), and it is drawn by its own script rather than by
experiment_detail.html's. Both mean the contract worth testing is the *payload*
— that the endpoint returns a skeleton whose prior and acquisition traces are
empty, and whose `layout.meta.acquisition` carries everything the browser needs
to fill them.

DB access + isolated media come from tests/ui/conftest.py.
"""

import json

import pytest
from pathlib import Path

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _slice(client, exp, hp, metric="accuracy"):
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": metric, "hp": hp})
    return response, (json.loads(response.content) if response.status_code == 200 else None)


def _panel(data, name):
    """One of the three figures. They are separate so each can carry its own
    controls; `meta` is shared because all three are drawn from the same
    numbers and it carries two thousand sampled points."""
    return data["figures"][name]


def _hp_names(exp):
    """Through the same rebuild the endpoint uses, so the names are the ones it
    will actually accept rather than a guess at the stored shape."""
    from ui.views import _hp_names as names_of, _rebuild_experiment

    built = _rebuild_experiment(exp)
    return names_of(built["result"]) if built and built["result"] else []


def test_the_figure_appears_on_the_page(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'data-figure="acquisition_slice"' in html
    assert 'id="figure-acquisition_slice"' in html
    # Its own script, deferred so it runs after the plotly tag further down.
    assert "ui/acquisition.js" in html
    assert "defer" in html


def test_the_page_announces_when_it_has_redrawn(client):
    """The figure draws itself, so the page purging it on a metric switch would
    otherwise wipe it at an unpredictable moment. One CustomEvent closes that,
    in the idiom `poll:swapped` already uses."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "figures:redrawn" in html


def _prior_trace_indices(meta, panel):
    """Every trace index the browser fills, flattened out of `meta.traces`.

    `fictionBands` holds pairs rather than a bare index, because a band is a
    lower and an upper trace with a fill between them.
    """
    indices = set()
    for key, value in (meta["traces"][panel] or {}).items():
        if key == "fictionBand":
            indices.update(value)
        else:
            indices.add(value)
    return indices


def test_every_named_trace_resolves_to_the_trace_it_names(client):
    """The browser addresses traces by index out of `meta.traces`, so the server
    can add bands and a marker without it knowing — provided the names still
    point where they claim. This is what keeps that true, and it replaces an
    assertion on the literal numbers 3-6, which the stepped bands moved."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    assert data.get("figures"), data.get("warning")

    meta = data["meta"]
    def name_at(panel, key):
        return _panel(data, panel)["data"][meta["traces"][panel][key]].get("name")

    assert name_at("surrogate", "fiction") == "Implied by the prior"
    assert name_at("prior", "prior") == "Prior"
    assert name_at("acquisition", "acquisition") == "Acquisition"
    assert name_at("acquisition", "weighted") == "Weighted by the prior"
    assert len(meta["traces"]["surrogate"]["fictionBand"]) == 2, "a lower and an upper"


def test_the_incumbent_is_marked(client):
    """One point on the slice was evaluated — every other hyperparameter is held
    at its value — and its distance from the predicted curve is the surrogate's
    error at the only place a reader can check it."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])

    marks = [t for t in _panel(data, "surrogate")["data"] if t.get("name") == "Incumbent"]
    assert len(marks) == 1
    assert len(marks[0]["x"]) == 1 and len(marks[0]["y"]) == 1
    assert marks[0]["mode"] == "markers"


def test_the_spread_is_stepped_rather_than_one_ribbon(client):
    """One ±2σ ribbon says "somewhere in here" and nothing about where the mass
    sits. Three nested bands say it at a glance — and the fiction gets the same
    treatment, so the two read as the same kind of object."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    filled = [t for t in _panel(data, "surrogate")["data"] if t.get("fill") == "tonexty"]
    # One per step for the surrogate, plus the fiction's single band.
    assert len(filled) == len(meta["sigmas"]) + 1
    assert meta["sigmas"] == sorted(meta["sigmas"], reverse=True), \
        "widest first, so the narrower and darker bands draw over them"
    assert meta["fictionSigma"] == 1


def test_the_fiction_does_not_bridge_its_own_gaps(client):
    """The inversion returns nothing where a prior claims more improvement than
    any finite mean could justify. Bridging that would draw a plateau the
    function deliberately refuses to invent."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    surrogate = _panel(data, "surrogate")["data"]
    fiction = ([meta["traces"]["surrogate"]["fiction"]]
               + list(meta["traces"]["surrogate"]["fictionBand"]))
    for i in fiction:
        assert surrogate[i].get("connectgaps") is False, i


def test_the_skeleton_leaves_the_prior_traces_empty(client):
    """The server draws what the run supports; the browser draws what depends on
    a prior it is holding. That division is visible in the payload as exactly
    the traces `meta.traces` names coming back empty — no more, so nothing the
    server drew is about to be overwritten, and no fewer, so nothing the browser
    owns was left filled.

    `candidates` is on the browser's side of the line for a second reason: it is
    not a redraw of anything, but what a real optimizer said it would run next,
    which only exists once the reader asks for it — and how many markers it
    takes depends on how wide the panel turned out to be."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response, data = _slice(client, exp, hp)

    assert response.status_code == 200
    assert data.get("figures"), data.get("warning")

    meta = data["meta"]
    for panel in ("acquisition", "prior", "surrogate"):
        empty = {i for i, trace in enumerate(_panel(data, panel)["data"])
                 if all(v is None for v in trace.get("y", [None]))}
        assert empty == _prior_trace_indices(meta, panel), panel

    assert set(meta["traces"]) == {"acquisition", "prior", "surrogate"}
    assert set(meta["traces"]["acquisition"]) == {"acquisition", "weighted", "cloud",
                                                 "candidates"}
    assert set(meta["traces"]["prior"]) == {"prior", "priorPoints"}
    assert set(meta["traces"]["surrogate"]) == {"fiction", "fictionBand"}


def test_the_payload_carries_what_the_browser_needs(client):
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    meta = data["meta"]

    assert len(meta["positions"]) == len(meta["mu"]) == len(meta["sigma"]) > 1
    assert isinstance(meta["higherIsBetter"], bool)
    assert meta["eta"] is not None
    assert meta["kind"] in {"continuous", "categorical"}
    assert meta["span"][0] < meta["span"][1]


def test_the_x_axis_is_pinned(client):
    """Our own pointer handler converts a pixel to a value from the panel's
    bounding box alone, which only holds while the axis cannot move. Plotly's
    own dragging is off for the same reason."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    span = list(data["meta"]["span"])

    # One axis each now, and the same one: separate figures share an x axis by
    # construction rather than by Plotly, since all three are given the same
    # span, ticks and positions.
    for panel in ("acquisition", "prior", "surrogate"):
        layout = _panel(data, panel)["layout"]
        assert layout["dragmode"] is False, panel
        assert layout["xaxis"]["fixedrange"] is True, panel
        assert list(layout["xaxis"]["range"]) == span, panel


def test_each_panel_names_only_its_own_traces(client):
    """One legend for all three panels puts names from three different pictures
    in a single strip, with nothing saying which panel a name belongs to. Each
    panel gets its own, and each legend sits above the panel it describes."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    # Each figure owns its own legend now. That used to need three legends on
    # one figure, positioned by hand over their panel's domain — the split does
    # it by construction, and a trace cannot land in another panel's strip.
    for panel in ("acquisition", "prior", "surrogate"):
        layout = _panel(data, panel)["layout"]
        named = [t for t in _panel(data, panel)["data"] if t.get("name")]
        assert named, f"{panel} has nothing to put in a legend"
        for trace in named:
            assert trace.get("legend") in (None, "legend"), (panel, trace["name"])
        assert layout["legend"]["yanchor"] == "top", panel
        assert layout["legend"]["y"] <= 1.0, panel
        # Inside its own figure, so it cannot land on another panel's title —
        # which is exactly what it did when all three shared one figure.
        assert layout["legend"]["xanchor"] == "right", panel



def test_every_hyperparameter_answers(client):
    exp = _experiment()
    for hp in _hp_names(exp):
        response, data = _slice(client, exp, hp)
        assert response.status_code == 200, hp
        assert data["figures"], (hp, data.get("warning"))
        assert set(data["figures"]) == {"acquisition", "prior", "surrogate"}, hp


def test_the_script_reveals_the_div_before_drawing_into_it():
    """The page's own `draw(key, null)` sets `el.hidden = true`, and this
    figure's payload is *always* null — it ships no server-rendered plot. Every
    other figure gets un-hidden by that same `draw` on the way past; this one
    bypasses it, so its script has to do the job itself.

    Getting this wrong has no visible failure mode beyond "nothing happens":
    the Compute prompt hides, `Plotly.react` runs without error, and the plot is
    drawn into a div nobody can see. It cost an afternoon once.

    Order matters as much as presence — a hidden element has no width, so Plotly
    would size the plot to nothing and keep that size once it was revealed.
    """
    source = (Path(__file__).parents[3] / "ui" / "static" / "ui" / "acquisition.js").read_text()

    reveal = source.index("plotEl.hidden = false")
    draw = source.index("Plotly.react(panels.acquisition")
    assert reveal < draw, "the div must be revealed before it is drawn into"


# ── the fit is shared across hyperparameters ─────────────────────────────────


def test_slicing_two_hyperparameters_fits_the_surrogate_once(client, monkeypatch):
    """The model depends on the metric and the trials, not on which
    hyperparameter is on the axis. Refitting per hyperparameter cost ~300ms on a
    Gaussian process and ~750ms on a forest at fifteen trials, paid again for
    every name in the picker.

    The fit *count* is the assertion, not the wall time — a timing test here
    would pass on a fast machine with the caching removed.
    """
    from core.optimizers.base import BaseOptimizer
    from ui import views

    views._SLICE_SURROGATES.clear()
    fits = []
    original = BaseOptimizer.slice_surrogate

    def counted(self, *args, **kwargs):
        fits.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BaseOptimizer, "slice_surrogate", counted)

    exp = _experiment()
    names = _hp_names(exp)[:2]
    assert len(names) == 2, "this experiment needs two hyperparameters to say anything"
    for hp in names:
        response, _ = _slice(client, exp, hp)
        assert response.status_code == 200

    assert len(fits) == 1, f"fitted {len(fits)} times for one (experiment, metric)"


def test_a_new_trial_invalidates_the_cached_fit(client, monkeypatch):
    """The trial count is in the key, which is what stops a live run drawing
    every later slice from the model it had at the beginning."""
    from core.optimizers.base import BaseOptimizer
    from ui import views

    views._SLICE_SURROGATES.clear()
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _slice(client, exp, hp)

    keys = list(views._SLICE_SURROGATES)
    assert len(keys) == 1
    assert exp.pk in keys[0] and "accuracy" in keys[0]
    # The trial count is one of the key's components, so a differing count
    # cannot resolve to this entry.
    counts = [k for k in keys[0] if isinstance(k, int) and k != exp.pk]
    assert counts, "no trial count in the cache key"


def _script():
    return (Path(__file__).parents[3] / "ui" / "static" / "ui" / "acquisition.js").read_text()


# ── stating a prior ─────────────────────────────────────────────────────────
#
# The prior lives entirely in the browser, and this project has no JavaScript
# test runner — no node, no npm, no package.json. So what follows asserts the
# wiring is present in the source, not that it behaves. The arithmetic these
# guard (`density`, `expectedImprovement`, `impliedMean`) is still executed by
# nothing in CI; adding a runner is the open decision, and it means taking on a
# Node toolchain this repo has so far done without.


def test_a_prior_is_stated_by_a_hand_rolled_pointer_gesture():
    """Plotly's shape editing cannot drive this and the reason is worth pinning.

    A shape drag repaints the SVG directly and reports only on mouse-up, so a
    curve recomputed from `plotly_relayout` would move once, at the end of the
    gesture. The drag layer is bound instead, which `dragmode=False` frees.
    """
    source = _script()

    assert "pointerdown" in source
    assert "setPointerCapture" in source, "the gesture must survive leaving the panel"
    assert "state.prior = {" in source, "a press with no prior must state one"


def test_the_prior_panel_is_found_by_its_y_axis():
    """`shared_xaxes` leaves the x half of the subplot id ambiguous ("xy2" or
    "x2y2"); y2 is the middle panel either way, so that is what is matched."""
    # It is its own figure now with a single subplot, so there is no ambiguity
    # left to match around: the div is the panel.
    assert 'panels.prior.querySelector(".nsewdrag")' in _script()


def test_the_drag_is_one_repaint_per_frame():
    """Trace `y` is `editType: "calc"`, so a restyle per trace would be four
    full recalcs per frame, and an uncoalesced pointer would queue recalcs that
    are stale before they run."""
    source = _script()

    assert "requestAnimationFrame" in source
    assert source.count("Plotly.update(") == 1, "all four traces go in one call"


def test_everything_on_the_acquisition_panel_shares_one_divisor():
    """Each curve to its own maximum made them incomparable: a prior that
    lifted the acquisition everywhere and one that flattened it drew the same
    picture, because each was rescaled to fill the panel.

    The weight is peak-normalized before the common divisor, which costs
    nothing — an acquisition function is only ever ranked, so a constant factor
    cannot change which configuration wins — and it is what stops a sharp
    prior's weight, which peaks in the hundreds, squashing the unweighted curve
    to invisibility.
    """
    source = _script()

    # The weight is still peak-normalized first, which is what stops a sharp
    # prior squashing the unweighted curve to invisibility.
    assert "peak = maxOf(weights)" in source

    # And the divisor is the largest of everything the panel draws, not the
    # curve's own maximum. The candidates outscore the slice by several times,
    # and the axis is pinned to [0, 1.08], so scaling to the curve put them off
    # the top of the figure rather than beside it.
    assert "if (maxOf(weighted) > scale) scale = maxOf(weighted);" in source
    assert "if (envelope && maxOf(envelope) > scale)" in source
    for divided in ("acq[i] /= scale;", "weighted[i] /= scale;",
                    "envelope[i] /= scale;"):
        assert divided in source, divided


def test_the_shadow_uses_the_surrogates_own_sigma():
    """A prior moves where the interest is; it does not claim to have narrowed
    the model's uncertainty. So the implied curve's shadow is drawn from the same
    σ the surrogate reports, at the same steps as the band above it."""
    source = _script()

    assert "meta.fictionSigma" in source
    assert "k * meta.sigma[i]" in source


def test_the_browser_never_counts_trace_indices():
    """The server adds bands and a marker conditionally, so the numbers move.
    Reading them from `meta.traces` is what makes that invisible here."""
    source = _script()

    assert "meta.traces" in source or "t.fictionBands" in source
    assert "priorTraces" in source, "one place that knows which traces a prior owns"


def test_a_prior_can_be_withdrawn():
    assert "dblclick" in _script()


def test_a_prior_outlives_a_metric_switch_but_not_a_new_hyperparameter():
    """The page redraws this figure from cache when the metric changes, which
    would otherwise drop a prior the reader had just dragged. A different
    hyperparameter is a different axis, where the same centre and width would
    mean something else, so there it is dropped."""
    source = _script()

    assert "state.hp === hp" in source


def test_an_unknown_metric_is_refused(client):
    exp = _experiment()
    response, _ = _slice(client, exp, _hp_names(exp)[0], metric="nonesuch")

    assert response.status_code == 400


def test_an_unknown_hyperparameter_is_refused(client):
    exp = _experiment()
    response, _ = _slice(client, exp, "nonesuch")

    assert response.status_code == 400


def test_an_experiment_with_no_result_does_not_500(client):
    """An empty experiment should say it has nothing, not raise."""
    from ui.models import Experiment

    exp = Experiment.objects.create(
        name="empty", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], current_metric="accuracy", seed=0)
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": "n_estimators"})

    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert json.loads(response.content)["figure"] is None


def _alpha(fillcolor):
    """The opacity out of an `rgba(r, g, b, a)` string."""
    return float(fillcolor.rsplit(",", 1)[1].strip(" )"))


def test_the_ghost_is_far_fainter_than_the_measured_spread(client):
    """The implied curve is an inference about a surrogate nobody fitted — read
    off an acquisition function by inverting it. One band rather than three, and
    a much fainter one, is what keeps it from reading as the model's own
    spread."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    surrogate = _panel(data, "surrogate")["data"]
    band = meta["traces"]["surrogate"]["fictionBand"][1]
    ghost = _alpha(surrogate[band]["fillcolor"])
    measured = [_alpha(t["fillcolor"]) for i, t in enumerate(surrogate)
                if t.get("fill") == "tonexty" and i != band]

    assert len(measured) == len(meta["sigmas"])
    assert ghost < min(measured) / 2, f"ghost {ghost} against {sorted(measured)}"


def test_every_panel_carries_the_hyperparameters_values(client):
    """`shared_xaxes` hides the upper panels' tick labels by default, which
    leaves a reader tracing a vertical line down three panels to work out which
    value they are looking at — and the top one is where the incumbent is
    marked."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    for panel in ("acquisition", "prior", "surrogate"):
        axis = _panel(data, panel)["layout"]["xaxis"]
        assert axis.get("showticklabels") is not False, panel
        assert axis.get("ticktext"), f"{panel} has no value labels"


# ── stating a prior by its parameters ────────────────────────────────────────


def test_the_page_offers_a_distribution_and_a_decay(client):
    """A single draggable bump is one shape. The picker is what makes the panel
    able to express a prior rather than only a Normal one."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-prior-kind"' in html
    assert 'id="acq-prior-decay"' in html
    assert 'id="acq-prior-params"' in html
    kinds = html[html.index('id="acq-prior-kind"'):html.index('id="acq-prior-decay"')]
    for kind in ('value="uniform"', 'value="normal"', 'value="beta"'):
        assert kind in kinds, kind


def test_uniform_is_the_resting_state_not_a_separate_no_prior(client):
    """A uniform prior multiplies the acquisition by a constant, which cannot
    change a ranking — so it says exactly what stating nothing says, and offering
    both would be two names for one thing."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    kinds = html[html.index('id="acq-prior-kind"'):html.index('id="acq-prior-decay"')]

    assert "No prior" not in kinds
    assert 'value="none"' not in kinds


def test_a_uniform_prior_weights_nothing_and_casts_no_ghost():
    """Under a uniform prior the implied curve would sit exactly on the predicted
    one by construction, which is clutter rather than information."""
    source = _script()

    assert 'prior.kind !== "uniform"' in source
    assert "fiction.push(stated" in source


def test_uniform_cannot_decay():
    """Decay shrinks a prior's exponent toward one as a run proceeds, and a
    uniform prior is already the constant it would decay to."""
    source = _script()

    # Disabled, not hidden: a control that vanishes takes its space with it and
    # the figure jumps under the reader's pointer.
    assert 'decaySelect.disabled = kind === "uniform"' in source


def test_the_parameters_are_labelled_with_their_symbols(client):
    """Mathematical names, not paraphrases. The symbol reads the same in every
    locale so it lives in the script; the prose description is translated in the
    template and shown as the control's tooltip."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    source = _script()

    assert '\u03bc' in source and '\u03c3' in source, "mu and sigma as symbols"
    assert '\u03b1' in source and '\u03b2' in source, "alpha and beta as symbols"
    assert 'id="acq-prior-strings"' in html
    strings = html[html.index('id="acq-prior-strings"'):]
    strings = strings[:strings.index("</div>")]
    assert 'data-mu="Mean"' in strings
    assert 'data-sigma="Standard deviation"' in strings
    assert "Centre" not in strings and "Width" not in strings


def test_the_browser_evaluates_no_density_of_its_own():
    """The one assertion that keeps there being a single implementation.

    The figure used to compute a prior's density in JavaScript so a drag could
    redraw at pointer speed, while `core.priors` computed it again for the
    search and for a prior restored from a file. Two Gaussians that had to
    agree forever, where a disagreement would not raise — it would weight the
    search by one shape while this figure drew another.

    Nothing needs it here now: the shapes with parameters are set through
    fields, which already round-trip, and the density comes back in the same
    response. Named as absences because that is what is being protected."""
    source = _script()

    assert "function density(" not in source
    assert "function through(" not in source, "no interpolation of a density here"
    # The shapes a reader can pick still live in one table; only their
    # evaluation moved.
    assert "var KINDS = {" in source, "one table, so a shape is an entry not a widget"
    assert "function densityFor(" in source, "it is read from the payload instead"


def test_the_density_arrives_with_the_payload(client):
    """And the figure draws from it rather than from anything it derived."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "beta", "params": {"alpha": 3, "beta": 2},
                            "decay": {"shape": "none"}})

    _response, data = _slice(client, exp, hp)

    assert len(data["density"]["grid"]) == len(data["meta"]["positions"])
    assert len(data["density"]["cloud"]) == len(data["meta"]["cloud"]["positions"])


def test_a_shape_with_fields_is_stated_only_in_its_fields():
    """There used to be two ways to set μ and σ: the numbers under the figure,
    and dragging the curve. The drag was the worse of them — less precise, and
    it named nothing — and having it there made every panel look editable, which
    invited dragging shapes that had nothing to give. So one writer, and the
    fields are it."""
    source = _script()

    assert "function syncControls()" in source
    assert "drag: {location:" not in source, "no shape is stated by dragging"
    # The parameter inputs write the parameters, and nothing else does.
    assert "function bindField(" in source
    assert "function syncFields()" not in source, "nothing left to sync back"


def test_only_the_freeform_prior_takes_a_pointer():
    """A Normal is two numbers that already have fields. A freeform prior is
    different in kind — its points *are* the statement, and no field could carry
    them — so it is the one shape the panel itself edits, and the one that
    offers a cursor."""
    source = _script()

    assert "function editable()" in source
    assert "KINDS[kind].points" in source
    assert 'cursor = editable() ? "crosshair" : "default"' in source


# ── freeform and categorical, one editor ─────────────────────────────────────


def test_freeform_is_offered_and_shares_the_categorical_editor(client):
    """"One weight per choice" and "a curve through points a reader placed" are
    the same object with a different constraint on x. Two editors for that would
    be two editors for one idea."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    source = _script()

    assert 'value="tabulated"' in html
    assert "tabulated: {params: [], points: true}" in source
    assert "function onTabulatedDown" in source
    assert 'currentMeta().kind === "categorical"' in source, \
        "the categorical case is a constraint on the same editor"


def test_the_control_points_are_their_own_trace(client):
    """The one trace whose x moves — a point is dragged along the axis as well
    as up it — so x travels with y instead of being fixed to the grid."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    points = _panel(data, "prior")["data"][meta["traces"]["prior"]["priorPoints"]]
    assert points["mode"] == "markers"
    assert points["x"] == [] and points["y"] == []

    source = _script()
    assert 'add("prior", t.prior.priorPoints, values.points.x, values.points.y)' in source
    assert "{x: group.xs, y: group.ys}" in source, "x and y in the same call"


def test_the_curve_through_the_points_cannot_dip_below_zero(client):
    """A density is non-negative. An ordinary cubic overshoots between points
    and would dip under zero on the way into a trough, inventing a region the
    reader wrote off when they did not.

    Asserted against the density the server returns, which is now the only one
    there is — `tests/core/test_prior_density.py` pins the interpolation itself.
    """
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp,
          {"kind": "tabulated",
           "params": {"points": [[0.0, 0.9], [0.45, 0.02], [1.0, 0.9]]},
           "decay": {"shape": "none"}})

    _response, data = _slice(client, exp, hp)

    assert all(y >= 0 for y in data["density"]["grid"])
    assert min(data["density"]["grid"]) <= 0.05


def test_a_hand_drawn_prior_gets_a_linear_panel():
    """A drawn height has no units and a log axis cannot reach zero, so the
    panel becomes a plain [0, 1] — which also keeps the drag pure geometry in
    both directions, as it already was in x."""
    source = _script()

    assert "function applyPriorAxis" in source
    # Its own figure, so its own first y axis rather than the second of three.
    assert '"yaxis.type"' in source


# ── a prior that outlives the page, and reaches the search ───────────────────


def _save(client, exp, hp, prior):
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps({"hp": hp, "prior": prior}),
                       content_type="application/json")


def test_a_stated_prior_is_kept(client):
    """A prior is not a way of looking at a run, it is a statement about the
    next one, so it has to outlive the page."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _save(client, exp, hp, {"kind": "normal",
                                       "params": {"mu": 0.8, "sigma": 0.1},
                                       "decay": {"shape": "none"}})
    exp.refresh_from_db()

    assert response.status_code == 200
    assert hp in exp.priors
    assert exp.priors[hp]["kind"] == "normal"
    assert exp.priors[hp]["params"] == {"mu": 0.8, "sigma": 0.1}
    # Four fields, and no evaluated curve. The density is computed where it is
    # needed, from these numbers, so a stored copy could only go stale.
    assert set(exp.priors[hp]) == {"kind", "params", "decay", "at_trial"}


def test_uniform_is_stored_as_nothing(client):
    """It multiplies the acquisition by a constant, so keeping a flat table
    would have the optimizer weight by something that cannot change a ranking."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                            "decay": {"shape": "none"}})

    _save(client, exp, hp, None)
    exp.refresh_from_db()

    assert hp not in exp.priors


def test_the_slice_hands_back_what_was_stated(client):
    """So the controls come back where the reader left them, without the figure
    having to ask separately."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "beta", "params": {"alpha": 3, "beta": 2},
                            "decay": {"shape": "linear"}})

    _response, data = _slice(client, exp, hp)

    assert data["prior"]["kind"] == "beta"
    assert data["prior"]["params"] == {"alpha": 3, "beta": 2}
    assert data["prior"]["decay"]["shape"] == "linear"


def test_a_density_cannot_be_posted_at_all(client):
    """There is no longer a curve in the request to validate, which is the
    point: an evaluated density was the one thing a caller could send that
    disagreed with the shape it claimed to be. Anything extra is ignored rather
    than stored."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _save(client, exp, hp,
                     {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                      "decay": {"shape": "none"},
                      "knots": [[0.0, -5.0], [1.0, 1.0]]})
    exp.refresh_from_db()

    assert response.status_code == 200
    assert "knots" not in exp.priors[hp]


def test_the_density_comes_back_with_the_save(client):
    """One round trip. The save is also the redraw, because the server is the
    only thing that evaluates a density now — so the curve the reader ends up
    looking at came from the same function the optimizer will weight by."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _response, data = _slice(client, exp, hp)
    meta = data["meta"]

    body = {"hp": hp, "rewalk": False,
            "prior": {"kind": "normal", "params": {"mu": 0.8, "sigma": 0.1},
                      "decay": {"shape": "none"}},
            "meta": {"positions": meta["positions"], "span": meta["span"],
                     "cloud": {"positions": meta["cloud"]["positions"]}}}
    out = json.loads(client.post(reverse("ui:save_prior", args=[exp.pk]),
                                 data=json.dumps(body),
                                 content_type="application/json").content)

    grid = out["density"]["grid"]
    assert len(grid) == len(meta["positions"])
    assert len(out["density"]["cloud"]) == len(meta["cloud"]["positions"])
    # And it is the density that was asked for, not just an array of the right
    # length: a Normal at 0.8 peaks at 0.8.
    peak = meta["positions"][grid.index(max(grid))]
    assert peak == pytest.approx(0.8, abs=2.0 / len(grid))


# ── the shadow: what the slice cannot see ────────────────────────────────────


def test_the_payload_carries_a_sample_of_the_whole_space(client):
    """The curve is a slice; SMAC's maximizer is not. The shadow needs
    predictions for configurations that move the hyperparameters this panel
    holds fixed, so the payload carries the sample rather than a curve."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    assert meta["cloud"] is not None, data.get("warning")
    cloud = meta["cloud"]
    assert len(cloud["positions"]) == len(cloud["mu"]) == len(cloud["sigma"])
    assert len(cloud["mu"]) > 100, "too few samples to estimate anything"


def test_the_server_sends_predictions_not_acquisition_values(client):
    """Expected improvement has one implementation, in the browser. Sending
    values rather than means and spreads would make a second one that has to
    agree with the first forever."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    cloud = data["meta"]["cloud"]

    assert set(cloud) == {"positions", "mu", "sigma"}


def test_the_shadow_is_a_maximum_per_slot_not_a_mean():
    """What matters is what the search could find at that value, not how the
    region scores on average."""
    source = _script()

    assert "if (envelope[slot] === null || value > envelope[slot])" in source
    assert "function cloudSlots" in source, "slots do not depend on the prior"


def test_the_shadow_shares_the_curves_divisor():
    """So the three can be read against each other — and so the shadow is
    allowed to exceed 1, which is the whole point it makes."""
    source = _script()

    assert "envelope[i] /= scale" in source


def test_the_shadow_is_named_for_what_it_is(client):
    """It is a maximum over a sample, so it is a slack lower bound on the true
    profile maximum. Measured on a four-hyperparameter run, 25x the samples
    moved it from 0.17 to 0.34 and never reached the slice's own peak."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    cloud = _panel(data, "acquisition")["data"][meta["traces"]["acquisition"]["cloud"]]
    assert cloud["name"] == "Best found by random sampling"


def test_the_controls_keep_their_space(client):
    """A control that appears and disappears moves the figure under the
    reader's pointer."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "acq-controls" in html and "acq-params" in html
    assert 'id="acq-prior-decay"' in html and "disabled" in html


def _has_weight_layer():
    try:
        import smac.acquisition.weight  # noqa: F401
    except ImportError:
        return False
    return True


def test_a_prior_that_cannot_bite_says_so(client):
    """On a SMAC without the acquisition weight layer a prior is drawn, saved,
    offered a re-walk, and ignored by both the search and the candidate walk.
    Nothing on the page would say so otherwise, and a belief that silently does
    nothing is worse than one that is refused."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                            "decay": {"shape": "none"}})

    _response, data = _slice(client, exp, hp)

    if _has_weight_layer():
        assert data["warning"] is None
    else:
        assert "acquisition weight layer" in (data["warning"] or "")


def test_nothing_is_claimed_when_no_prior_is_stated(client):
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])

    assert data["warning"] is None


def test_a_normal_may_be_as_wide_as_the_axis():
    """Capping sigma below the axis width stops a reader expressing "nearly
    flat, with a slight tilt". Beyond the width it is uniform anyway."""
    source = _script()

    assert "min: 0.004, max: 1.0, step: 0.002" in source


# ── the prior fades, and says so ─────────────────────────────────────────────

def _state(client, exp, hp, kind="normal", decay="logarithmic"):
    """State a prior through the endpoint, so `at_trial` is recorded the way a
    reader stating one would record it."""
    body = {"hp": hp, "rewalk": False,
            "prior": {"kind": kind, "params": {"mu": 0.6, "sigma": 0.2},
                      "decay": {"shape": decay}}}
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps(body), content_type="application/json")


def test_a_stated_prior_weakens_as_the_run_goes_on():
    """The point of a decay, and the one thing about it a reader can check by
    watching. It ran backwards once: the decay factor was taken from the count
    of *finished* trials, so it grew alongside the step count it divides, and a
    prior drawn decaying got steadily stronger instead. Hence a direction
    assertion rather than a value one — the values are SMAC's."""
    from ui.views import _decayed_prior, _rebuild_experiment

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuild_experiment(exp)["result"].trials)
    exp.priors = {hp: {"kind": "normal", "params": {}, "decay": "logarithmic",
                       "at_trial": at}}
    exp.save(update_fields=["priors"])

    seen = [_decayed_prior(exp, hp, at + n)["exponent"] for n in (0, 1, 5, 20, 100)]

    assert seen == sorted(seen, reverse=True), seen
    assert seen[-1] < seen[0]


def test_a_prior_that_does_not_decay_keeps_its_strength():
    """`none` is a real choice and the default. Pinned separately because a
    schedule that quietly decayed anyway would still pass the test above."""
    from ui.views import _decayed_prior, _rebuild_experiment

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuild_experiment(exp)["result"].trials)
    exp.priors = {hp: {"kind": "normal", "params": {}, "decay": "none",
                       "at_trial": at}}
    exp.save(update_fields=["priors"])

    assert {_decayed_prior(exp, hp, at + n)["exponent"] for n in (0, 5, 100)} == {1.0}


def test_the_figure_ships_the_exponent_that_currently_applies(client):
    """The browser does not compute this. Six decay curves reimplemented there
    would be a second definition to keep in step with SMAC's."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    assert _state(client, exp, hp).status_code == 200

    _, data = _slice(client, exp, hp)

    assert isinstance(data["prior"]["exponent"], float)
    assert data["prior"]["decay"]["shape"] == "logarithmic"


# ── what the optimizer would actually do next ────────────────────────────────

def _smac_experiment():
    """The SMAC fixture rather than the random-search one the rest of this file
    uses. Only a run with a surrogate and an acquisition function has anything
    to rank."""
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test.ihpo").read_bytes()))


def test_asking_the_optimizer_returns_a_ranked_slate(client):
    """Not a lookalike. `slice_challengers` builds this run's own facade, replays
    its trials into it and calls `ask` — so what comes back is what would run,
    and the origins are SMAC's own words for how it got there."""
    exp = _smac_experiment()
    hp = _hp_names(exp)[0]

    body = {"hp": hp, "prior": None, "rewalk": True}
    response = client.post(reverse("ui:save_prior", args=[exp.pk]),
                           data=json.dumps(body), content_type="application/json")
    candidates = json.loads(response.content)["candidates"]

    assert response.status_code == 200
    assert candidates
    # Ranks are a display number: 1..n, no gaps, best first.
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    for c in candidates:
        assert 0.0 <= c["position"] <= 1.0
        assert c["origin"]
        # The whole configuration, not just the coordinate this axis shows —
        # a candidate differs from the incumbent in every dimension.
        assert set(c["config"]) == set(_hp_names(exp))
        assert all(f"{k} =" in c["label"] for k in c["config"])


def test_not_asking_costs_nothing(client):
    """A model fit and a maximization per keystroke would be the cost of doing
    this on every edit. It is a button for that reason."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _state(client, exp, hp)

    assert response.status_code == 200
    assert json.loads(response.content)["candidates"] is None


def test_an_optimizer_with_nothing_to_rank_says_so_rather_than_failing(client):
    """Random and grid search have no surrogate and no acquisition function, so
    there is no next candidate to report. Asking one used to raise
    `AttributeError` and return a 500 to a reader who had done nothing wrong."""
    exp = _experiment()  # random search
    hp = _hp_names(exp)[0]

    body = {"hp": hp, "prior": None, "rewalk": True}
    response = client.post(reverse("ui:save_prior", args=[exp.pk]),
                           data=json.dumps(body), content_type="application/json")

    assert response.status_code == 200
    assert json.loads(response.content)["candidates"] is None


# ── the phase a prior cannot reach ───────────────────────────────────────────

def test_a_run_still_in_its_initial_design_says_so(client):
    """The initial design is drawn before a surrogate exists, so a prior stated
    while it is running cannot touch the trials still to come out of it. In a
    probe the first six trials were identical with and without a prior for
    exactly this reason."""
    from ui.views import _initial_design_notice, _rebuild_experiment

    exp = _smac_experiment()
    result = _rebuild_experiment(exp)["result"]

    class StillGoing:
        metadata = result.metadata
        trials = result.trials[:1]

    notice = _initial_design_notice(StillGoing)

    assert notice["done"] is False
    assert notice["trials"] == 1
    assert notice["size"] >= 1


def test_a_finished_design_raises_no_banner(client):
    """Once the design is exhausted a prior reaches every trial, so the banner
    would be saying something untrue."""
    from ui.views import _initial_design_notice, _rebuild_experiment

    result = _rebuild_experiment(_smac_experiment())["result"]

    class Finished:
        metadata = dict(result.metadata,
                        initial_design={"name": "SobolInitialDesign", "n_configs": 1,
                                        "additional_configs": []})
        trials = result.trials

    assert _initial_design_notice(Finished)["done"] is True


def test_the_blackbox_fixture_never_left_its_initial_design(client):
    """Not a contrived case — the fixture is one. Its scenario asked for 32
    initial points and the run recorded 30 trials, so every trial in it was
    drawn before a surrogate existed. Worth pinning: it means a prior stated
    against this run would reach nothing at all, and the banner is the only
    thing on the page that would say so."""
    from ui.views import _initial_design_notice, _rebuild_experiment

    notice = _initial_design_notice(_rebuild_experiment(_smac_experiment())["result"])

    assert notice["size"] == 32
    assert notice["trials"] == 30
    assert notice["done"] is False


def test_the_default_configuration_counts_toward_the_phase(client):
    """`use_default_config` pins SMAC's default into `additional_configs`, which
    sits *outside* `n_configs` while being drawn in the same model-free phase.
    Left out, the banner would clear one trial early."""
    from ui.views import _initial_design_notice, _rebuild_experiment

    result = _rebuild_experiment(_smac_experiment())["result"]

    class WithDefault:
        metadata = dict(result.metadata,
                        initial_design={"name": "SobolInitialDesign", "n_configs": 3,
                                        "additional_configs": [{"x": 1}]})
        trials = result.trials

    assert _initial_design_notice(WithDefault)["size"] == 4


def test_the_size_is_read_and_never_guessed(client):
    """No recorded size means no banner. Counting origins instead is forbidden
    by `tests/core/test_initial_points.py` — the default configuration carries
    an initial-design origin while sitting outside `n_configs`, and old files
    label every trial with the optimizer's name."""
    from ui.views import _initial_design_notice, _rebuild_experiment

    result = _rebuild_experiment(_smac_experiment())["result"]

    class Bare:
        metadata = {}
        trials = result.trials

    assert _initial_design_notice(Bare) is None
    # Random search has no recorded design at all, through the real endpoint.
    exp = _experiment()
    _, data = _slice(client, exp, _hp_names(exp)[0])
    assert data["initialDesign"] is None


def test_the_figure_carries_the_banner_element(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-initial"' in html


# ── the decay factor is stated, not derived ──────────────────────────────────

def _state_beta(client, exp, hp, ratio, shape="quadratic"):
    body = {"hp": hp, "rewalk": False,
            "prior": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                      "decay": {"shape": shape, "beta_ratio": ratio}}}
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps(body), content_type="application/json")


def test_beta_is_stored_as_a_ratio_of_the_budget(client):
    """πBO states β as N/10 and DynaBO keeps that, ablating N/50 … N/2.5
    (Appendix F.9) — a fraction of the trial budget, never an absolute. Stored
    as the ratio so it survives a change of budget meaning the same thing: a
    prior exported from a 50-trial run keeps its strength relative to whatever
    run it is reopened in."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    assert _state_beta(client, exp, hp, 0.2).status_code == 200
    exp.refresh_from_db()

    assert exp.priors[hp]["decay"] == {"shape": "quadratic", "beta_ratio": 0.2}


def test_the_ratio_resolves_against_the_budget(client):
    """β itself is the ratio times N, and it is resolved rather than stored —
    which is the whole point of keeping the ratio."""
    from ui.views import _decayed_prior, _rebuild_experiment, _run_budget

    exp = _experiment()
    hp = _hp_names(exp)[0]
    trials = len(_rebuild_experiment(exp)["result"].trials)
    _state_beta(client, exp, hp, 0.2)
    exp.refresh_from_db()

    out = _decayed_prior(exp, hp, trials)
    at = exp.priors[hp]["at_trial"]

    assert out["beta"] == pytest.approx(_run_budget(exp, at) * 0.2)


def test_the_ratio_is_held_inside_the_ablated_range(client):
    """The bounds are DynaBO's own grid opened a little at each end. Narrower
    would refuse a value the paper measured; a ratio near one leaves the
    exponent above one for most of the run, which is a prior that never
    fades."""
    from ui.views import BETA_RATIO_MAX, BETA_RATIO_MIN

    exp = _experiment()
    hp = _hp_names(exp)[0]
    for asked, expected in ((-5.0, BETA_RATIO_MIN), (100.0, BETA_RATIO_MAX)):
        _state_beta(client, exp, hp, asked)
        exp.refresh_from_db()
        assert exp.priors[hp]["decay"]["beta_ratio"] == expected


def test_an_absolute_beta_is_read_as_the_ratio_it_was(client):
    """Priors stated before β became a ratio recorded it absolutely. Dropping
    them would lose a stated prior on upgrade, and reading the number as a
    ratio would silently make it about fifty times weaker."""
    from ui.views import _decayed_prior, _rebuild_experiment, _run_budget

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuild_experiment(exp)["result"].trials)
    budget = _run_budget(exp, at)
    exp.priors = {hp: {"kind": "normal", "params": {}, "at_trial": at,
                       "decay": {"shape": "linear", "beta": budget * 0.2}}}
    exp.save(update_fields=["priors"])

    out = _decayed_prior(exp, hp, at)

    assert out["decay"]["beta_ratio"] == pytest.approx(0.2)


def test_an_old_flat_decay_string_still_loads(client):
    """Priors stated before β was settable recorded the shape as a bare string.
    Refusing them would lose a stated prior on upgrade."""
    from ui.views import _decayed_prior, _rebuild_experiment

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuild_experiment(exp)["result"].trials)
    exp.priors = {hp: {"kind": "normal", "params": {}, "decay": "linear",
                       "at_trial": at}}
    exp.save(update_fields=["priors"])

    out = _decayed_prior(exp, hp, at + 5)

    assert out["decay"]["shape"] == "linear"
    assert out["decay"]["beta_ratio"] > 0
    assert out["exponent"] > 0
