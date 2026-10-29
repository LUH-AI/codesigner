"""`acquisition.js` draws the acquisition figure itself, so the tests pin the
endpoint's payload: empty prior and acquisition traces, plus a `meta` carrying
what the browser needs to fill them.
"""

import json
from typing import Any

import pytest
from pathlib import Path

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment():
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _slice(client, exp, hp, metric="accuracy") -> tuple[Any, Any]:
    """`_slice` returns the response and its JSON, with built but unsent figures
    added under `figures` and the sent names under `shipped`.
    """
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": metric, "hp": hp})
    if response.status_code != 200:
        return response, None
    data = json.loads(response.content)
    if isinstance(data.get("figures"), dict):
        data["shipped"] = set(data["figures"])
        # Sent figures win, so shipped-figure assertions test the arrived bytes.
        data["figures"] = dict(_built_figures(exp, hp, metric), **data["figures"])
    return response, data


def _built_figures(exp, hp, metric="accuracy"):
    """`_built_figures` builds every slice figure through the endpoint's seam."""
    from experiments.services.snapshot import config_space_for, rebuild_experiment
    from ui.views.common import _plot_json
    from ui.views.priors import _slice_figures

    built = rebuild_experiment(exp)
    figures, _meta, _sliced = _slice_figures(
        exp, built, config_space_for(built), hp, metric)
    return {name: _plot_json(fig) for name, fig in (figures or {}).items()}


def _panel(data, name):
    """`_panel` returns one figure, and the three share one `meta` because it
    carries two thousand sampled points."""
    return data["figures"][name]


def _hp_names(exp):
    """`_hp_names` uses the endpoint's rebuild, so the endpoint accepts every
    name."""
    from experiments.services.snapshot import rebuild_experiment
    from ui.views.common import _hp_names as names_of

    built = rebuild_experiment(exp)
    return names_of(built["result"]) if built and built["result"] else []


def _rebuilt_result(exp):
    """`_rebuilt_result` rebuilds the stored result like the endpoint."""
    from experiments.services.snapshot import rebuild_experiment

    built = rebuild_experiment(exp)
    assert built is not None
    return built["result"]


def _decayed(exp, hp, trials):
    from experiments.services.priors import decayed_prior

    out = decayed_prior(exp, hp, trials)
    assert out is not None
    return out


def test_the_figure_appears_on_the_page(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'data-figure="acquisition_slice"' in html
    assert 'id="figure-acquisition_slice"' in html
    # Deferred so it runs after the plotly tag.
    assert "ui/acquisition.js" in html
    assert "defer" in html


def test_the_page_announces_when_it_has_redrawn(client):
    """The page purges self-drawn figures on a metric switch, so
    `figures:redrawn` tells `acquisition.js` to redraw."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "figures:redrawn" in html


def _prior_trace_indices(meta, panel):
    """`_prior_trace_indices` flattens `meta.traces`, where `fictionBand` contains
    a lower and upper pair.
    """
    indices = set()
    for key, value in (meta["traces"][panel] or {}).items():
        if key == "fictionBand":
            indices.update(value)
        else:
            indices.add(value)
    return indices


def test_every_named_trace_resolves_to_the_trace_it_names(client):
    """The browser finds traces through `meta.traces`, so each name must point
    at its trace."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    assert data.get("figures"), data.get("warning")

    meta = data["meta"]
    def name_at(panel, key):
        return _panel(data, panel)["data"][meta["traces"][panel][key]].get("name")

    # The names are translatable, so the test checks keys and never wording.
    for panel in ("acquisition", "prior", "surrogate"):
        keys = [k for k in meta["traces"][panel] if k != "fictionBand"]
        seen = [meta["traces"][panel][k] for k in keys]
        assert len(set(seen)) == len(seen), (panel, "two keys, one trace")
        for key in keys:
            assert name_at(panel, key), (panel, key, "no legend entry")
    assert name_at("acquisition", "acquisition") != name_at("acquisition", "weighted")
    dashed = _panel(data, "acquisition")["data"][meta["traces"]["acquisition"]["acquisition"]]
    assert dashed["line"]["dash"] == "dash", "the unweighted one is the dashed one"
    assert len(meta["traces"]["surrogate"]["fictionBand"]) == 2, "a lower and an upper"


def test_the_incumbent_is_marked(client):
    """The incumbent is the slice's only measured point, so its gap to the curve
    shows the surrogate's error."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])

    marks = [t for t in _panel(data, "surrogate")["data"] if t.get("name") == "Incumbent"]
    assert len(marks) == 1
    assert len(marks[0]["x"]) == 1 and len(marks[0]["y"]) == 1
    assert marks[0]["mode"] == "markers"


def test_the_spread_is_stepped_rather_than_one_ribbon(client):
    """The surrogate panel draws one filled band per sigma, nested widest first,
    instead of one ±2σ ribbon."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    filled = [t for t in _panel(data, "surrogate")["data"] if t.get("fill") == "tonexty"]
    # One band per sigma, plus the fiction's.
    assert len(filled) == len(meta["sigmas"]) + 1
    assert meta["sigmas"] == sorted(meta["sigmas"], reverse=True), \
        "widest first, so the narrower and darker bands draw over them"
    assert meta["fictionSigma"] == 1


def test_the_fiction_does_not_bridge_its_own_gaps(client):
    """`impliedMean` returns null where no finite mean fits, and bridging would
    invent a plateau."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    surrogate = _panel(data, "surrogate")["data"]
    fiction = ([meta["traces"]["surrogate"]["fiction"]]
               + list(meta["traces"]["surrogate"]["fictionBand"]))
    for i in fiction:
        assert surrogate[i].get("connectgaps") is False, i


def test_the_skeleton_leaves_the_prior_traces_empty(client):
    """The browser draws whatever depends on its prior, so the traces
    `meta.traces` names, and only those, arrive empty. `candidates` contains the
    optimizer's next trials, which only exist once the reader asks."""
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
                                                 "cloudBare", "candidates",
                                                 # `incumbent` is filled and
                                                 # cleared with the slice
                                                 # curves.
                                                 "incumbent"}
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
    """The pointer handler maps pixels to values from the bounding box alone,
    which needs a fixed axis."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    span = list(data["meta"]["span"])

    # Separate figures share an x axis by construction, with one span, ticks and
    # positions.
    for panel in ("acquisition", "prior", "surrogate"):
        layout = _panel(data, panel)["layout"]
        assert layout["dragmode"] is False, panel
        assert layout["xaxis"]["fixedrange"] is True, panel
        assert list(layout["xaxis"]["range"]) == span, panel


def test_each_panel_names_only_its_own_traces(client):
    """Each panel's legend sits above it, so no strip mixes three pictures."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    for panel in ("acquisition", "prior", "surrogate"):
        layout = _panel(data, panel)["layout"]
        named = [t for t in _panel(data, panel)["data"] if t.get("name")]
        assert named, f"{panel} has nothing to put in a legend"
        for trace in named:
            assert trace.get("legend") in (None, "legend"), (panel, trace["name"])
        # Inside the top-right corner the legend covered a sharp prior's peak.
        assert layout["legend"]["yanchor"] == "bottom", panel
        assert layout["legend"]["y"] > 1, panel
        assert layout["margin"]["t"] >= 40, (panel, "no room for the legend")
        assert layout["legend"]["xanchor"] == "right", panel



def test_every_hyperparameter_answers(client):
    exp = _experiment()
    for hp in _hp_names(exp):
        response, data = _slice(client, exp, hp)
        assert response.status_code == 200, hp
        assert data["figures"], (hp, data.get("warning"))
        assert set(data["figures"]) == {"acquisition", "prior", "surrogate"}, hp


def test_the_script_reveals_the_div_before_drawing_into_it():
    """`draw(key, null)` hides the div and the acquisition payload is always
    null, so `acquisition.js` unhides the div before drawing, since Plotly sizes
    a hidden element to zero.
    """
    source = (Path(__file__).parents[3] / "ui" / "static" / "ui" / "acquisition.js").read_text()

    reveal = source.index("plotEl.hidden = false")
    draw = source.index("Plotly.react(panels.acquisition")
    assert reveal < draw, "the div must be revealed before it is drawn into"


# Surrogate fit caching


def test_slicing_two_hyperparameters_fits_the_surrogate_once(client, monkeypatch):
    """The surrogate only depends on the metric and trials. The test counts
    fits, since a timing check would pass on a fast machine without the cache.
    """
    from core.optimizers.base import BaseOptimizer
    from ui.views import priors

    priors._SLICE_SURROGATES.clear()
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
    """The trial count in the cache key keeps a live run from slicing an
    outdated model."""
    from core.optimizers.base import BaseOptimizer
    from ui.views import priors

    priors._SLICE_SURROGATES.clear()
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _slice(client, exp, hp)

    keys = list(priors._SLICE_SURROGATES)
    assert len(keys) == 1
    assert exp.pk in keys[0] and "accuracy" in keys[0]
    # A differing trial count cannot resolve to this entry.
    counts = [k for k in keys[0] if isinstance(k, int) and k != exp.pk]
    assert counts, "no trial count in the cache key"


def _script():
    return (Path(__file__).parents[3] / "ui" / "static" / "ui" / "acquisition.js").read_text()


# Acquisition panel

def test_the_incumbent_is_marked_along_the_acquisition_curve(client):
    """The payload carries the incumbent's position in `meta` and an empty trace
    for its marker, whose height the browser computes.
    """
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response, data = _slice(client, exp, hp)

    assert response.status_code == 200
    meta = data["meta"]

    assert meta["incumbent"] is not None, "nowhere to put the marker"
    assert len(meta["incumbent"]) == 2, "a position and the score beside it"
    index = meta["traces"]["acquisition"]["incumbent"]
    assert _panel(data, "acquisition")["data"][index]["x"] == []


def test_the_prior_only_payload_has_no_incumbent(client):
    """Only the fit places the incumbent, and the prior-only payload fits no
    model."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": hp, "prior_only": "1"})
    meta = json.loads(response.content)["meta"]

    assert meta.get("incumbent") is None


def test_the_slice_curves_start_switched_off(client):
    """The slice curves start off because they answer a narrower question than
    the sampled envelope."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    at = html.index('id="acq-show-slice"')

    assert "checked" not in html[at:at + 60]


def test_its_controls_wait_for_the_curve(client):
    """Both controls stay hidden until the script draws the acquisition curve
    they depend on.
    """
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    card = html[html.index('data-figure="acquisition_slice"'):]

    for control in ('id="acq-show-slice"', 'id="acq-next-trials"'):
        at = card.index(control)
        opening = card.rindex("data-slice-control", 0, at)
        tag = card[opening:card.index(">", opening)]
        assert "hidden" in tag, control
        assert card.index('id="acq-hp-select"') < at, control
    assert "Acq. w/ inc. HPs" in card


def test_a_decaying_prior_says_how_much_of_it_is_left(client):
    """The readout is a template string, so translators reach the wording."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-prior-strength"' in html
    assert "data-strength=" in html, "the wording belongs to the template"
    assert "showStrength" in _script(), "and something has to fill it"


def test_the_knots_are_drawn_where_the_curve_is(client):
    """The markers and the hit test both pass through `decayed`, so knots sit on
    the weighted curve and get grabbed where drawn."""
    source = _script()

    assert "pts.y.push(decayed(" in source, "markers follow the exponent"
    assert "decayed(points[i][1], decayExponent())" in source, "and so does the grab"
    assert "stated(Math.min(" in source, "and a drop converts back"


def test_the_payload_ships_only_what_is_drawn(client):
    """The surrogate figure is built but never sent, saving a few thousand
    points per request."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response, data = _slice(client, exp, hp)

    assert response.status_code == 200
    assert data["shipped"] == {"acquisition", "prior"}
    assert "surrogate" in _built_figures(exp, hp), "still built, just not sent"


def test_the_surrogate_panel_is_not_on_the_page(client):
    """The surrogate figure is built but unsent, and the template has no div for
    it."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-plot-surrogate"' not in html
    assert 'id="acq-plot-acquisition"' in html, "the other two are still drawn"
    assert 'id="acq-plot-prior"' in html


# `acquisition.js` has no JavaScript test runner behind it, so the source checks
# below cover only its wiring.


def test_a_prior_is_stated_by_a_hand_rolled_pointer_gesture():
    """The script binds the drag layer that `dragmode=False` frees, because
    Plotly reports a shape drag only on mouse-up.
    """
    source = _script()

    assert "pointerdown" in source
    assert "setPointerCapture" in source, "the gesture must survive leaving the panel"
    assert "state.prior = {" in source, "a press with no prior must state one"


def test_the_prior_panel_is_found_by_its_y_axis():
    """`priorLayer` finds the drag layer inside the prior panel's own div."""
    assert 'panels.prior.querySelector(".nsewdrag")' in _script()


def test_the_drag_is_one_repaint_per_frame():
    """Trace `y` has `editType: "calc"`, so per-trace restyles or uncoalesced
    pointer events would queue full recalcs."""
    source = _script()

    assert "requestAnimationFrame" in source
    assert source.count("Plotly.update(") == 1, "all four traces go in one call"


def test_everything_on_the_acquisition_panel_shares_one_divisor():
    """Each curve scaled to its own maximum made a lifting prior and a
    flattening prior look the same. Peak-normalizing the weight first keeps a
    sharp prior from flattening the unweighted curve.
    """
    source = _script()

    assert "peak = maxOf(weights)" in source

    # The divisor is the panel's largest value because the candidates outscore
    # the slice several times over.
    assert "if (maxOf(weighted) > scale) scale = maxOf(weighted);" in source
    assert "if (envelope && maxOf(envelope) > scale)" in source
    for divided in ("acq[i] /= scale;", "weighted[i] /= scale;",
                    "envelope[i] /= scale;"):
        assert divided in source, divided


def test_the_shadow_uses_the_surrogates_own_sigma():
    """A prior leaves the model's uncertainty alone, so the shadow uses the
    surrogate's own σ."""
    source = _script()

    assert "meta.fictionSigma" in source
    assert "k * meta.sigma[i]" in source


def test_the_browser_never_counts_trace_indices():
    source = _script()

    assert "meta.traces" in source or "t.fictionBands" in source
    assert "priorTraces" in source, "one place that knows which traces a prior owns"


def test_a_prior_can_be_withdrawn():
    assert "dblclick" in _script()


def test_a_prior_outlives_a_metric_switch_but_not_a_new_hyperparameter():
    """The prior survives a metric switch, and a new hyperparameter drops it
    because the axis changes."""
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
    from experiments.models import Experiment

    exp = Experiment.objects.create(
        name="empty", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], current_metric="accuracy", seed=0)
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": "n_estimators"})

    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert json.loads(response.content)["figure"] is None


def _alpha(fillcolor):
    """`_alpha` reads an `rgba(...)` string's alpha."""
    return float(fillcolor.rsplit(",", 1)[1].strip(" )"))


def test_the_ghost_is_far_fainter_than_the_measured_spread(client):
    """The ghost gets a single faint band because the implied curve is an
    inference, never the model's own spread."""
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
    """Each panel labels its own x ticks."""
    exp = _experiment()
    response, data = _slice(client, exp, _hp_names(exp)[0])
    for panel in ("acquisition", "prior", "surrogate"):
        axis = _panel(data, panel)["layout"]["xaxis"]
        assert axis.get("showticklabels") is not False, panel
        assert axis.get("ticktext"), f"{panel} has no value labels"


# Prior parameters


def test_the_page_offers_a_distribution_and_a_decay(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-prior-kind"' in html
    assert 'id="acq-prior-decay"' in html
    assert 'id="acq-prior-params"' in html
    kinds = html[html.index('id="acq-prior-kind"'):html.index('id="acq-prior-decay"')]
    for kind in ('value="uniform"', 'value="normal"', 'value="beta"'):
        assert kind in kinds, kind


def test_uniform_is_the_resting_state_not_a_separate_no_prior(client):
    """A uniform prior leaves the ranking unchanged, so it already says what
    stating nothing says."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    kinds = html[html.index('id="acq-prior-kind"'):html.index('id="acq-prior-decay"')]

    assert "No prior" not in kinds
    assert 'value="none"' not in kinds


def test_a_uniform_prior_weights_nothing_and_casts_no_ghost():
    source = _script()

    assert 'prior.kind !== "uniform"' in source
    assert "fiction.push(stated" in source


def test_uniform_cannot_decay():
    """Decay shrinks a prior's exponent toward one, and a uniform prior is
    already that constant."""
    source = _script()

    # A disabled select keeps its space, so the figure stays under the pointer.
    assert 'decaySelect.disabled = kind === "uniform"' in source


def test_the_parameters_are_labelled_with_their_symbols(client):
    """Symbols live in the script, and the template translates the prose
    tooltip."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    source = _script()

    assert '\u03bc' in source and '\u03c3' in source, "mu and sigma as symbols"
    assert '\u03b1' in source and '\u03b2' in source, "alpha and beta as symbols"
    assert "data-acq-prior-strings" in html
    strings = html[html.index("data-acq-prior-strings"):]
    strings = strings[:strings.index("</div>")]
    assert 'data-mu="Mean"' in strings
    assert 'data-sigma="Standard deviation"' in strings
    assert "Centre" not in strings and "Width" not in strings


def test_the_browser_evaluates_no_density_of_its_own():
    """Two density implementations could disagree without raising, so the server
    alone computes the density."""
    source = _script()

    assert "function density(" not in source
    assert "function through(" not in source, "no interpolation of a density here"
    assert "var KINDS = {" in source, "one table, so a shape is an entry not a widget"
    assert "function densityFor(" in source, "it is read from the payload instead"


def test_the_density_arrives_with_the_payload(client):
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "beta", "params": {"alpha": 3, "beta": 2},
                            "decay": {"shape": "none"}})

    _response, data = _slice(client, exp, hp)

    assert len(data["density"]["grid"]) == len(data["meta"]["positions"])
    assert len(data["density"]["cloud"]) == len(data["meta"]["cloud"]["positions"])


def test_a_shape_with_fields_is_stated_only_in_its_fields():
    """Fields alone set μ and σ, since dragging was imprecise and made every
    panel look editable."""
    source = _script()

    assert "function syncControls()" in source
    assert "drag: {location:" not in source, "no shape is stated by dragging"
    assert "function bindField(" in source
    assert "function syncFields()" not in source, "nothing left to sync back"


def test_only_the_freeform_prior_takes_a_pointer():
    """A freeform prior's points fit no field, so freeform is the one shape the
    panel edits."""
    source = _script()

    assert "function editable()" in source
    assert "KINDS[kind].points" in source
    assert 'cursor = editable() ? "crosshair" : "default"' in source


# Freeform and categorical editor


def test_freeform_is_offered_and_shares_the_categorical_editor(client):
    """Choice weights and a curve through points only differ in their x
    constraint."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    source = _script()

    assert 'value="tabulated"' in html
    assert "tabulated: {params: [], points: true}" in source
    assert "function onTabulatedDown" in source
    assert 'currentMeta().kind === "categorical"' in source, \
        "the categorical case is a constraint on the same editor"


def test_the_control_points_are_their_own_trace(client):
    """The control points are the one trace whose x moves along with its y
    values."""
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
    """An ordinary cubic would overshoot below zero going into a trough.
    `tests/core/test_prior_density.py` pins the interpolation itself.
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
    """A drawn height has no units and a log axis cannot reach zero."""
    source = _script()

    assert "function applyPriorAxis" in source
    assert '"yaxis.type"' in source


# Saved priors


def _save(client, exp, hp, prior):
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps({"hp": hp, "prior": prior}),
                       content_type="application/json")


def test_a_stated_prior_is_kept(client):
    """A prior has to outlive the page because it states something about the
    next run."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _save(client, exp, hp, {"kind": "normal",
                                       "params": {"mu": 0.8, "sigma": 0.1},
                                       "decay": {"shape": "none"}})
    exp.refresh_from_db()

    assert response.status_code == 200
    assert hp in exp.data.priors
    assert exp.data.priors[hp]["kind"] == "normal"
    assert exp.data.priors[hp]["params"] == {"mu": 0.8, "sigma": 0.1}
    # `exp.data.priors` contains no evaluated curve that could go stale, and
    # `delay_decay` derives from nothing else.
    assert set(exp.data.priors[hp]) == {"kind", "params", "decay", "at_trial",
                                   "delay_decay"}


def test_uniform_is_stored_as_nothing(client):
    """A flat table would weight the optimizer by a constant that cannot change
    a ranking."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                            "decay": {"shape": "none"}})

    _save(client, exp, hp, None)
    exp.refresh_from_db()

    assert hp not in exp.data.priors


def test_the_slice_hands_back_what_was_stated(client):
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _save(client, exp, hp, {"kind": "beta", "params": {"alpha": 3, "beta": 2},
                            "decay": {"shape": "linear"}})

    _response, data = _slice(client, exp, hp)

    assert data["prior"]["kind"] == "beta"
    assert data["prior"]["params"] == {"alpha": 3, "beta": 2}
    assert data["prior"]["decay"]["shape"] == "linear"


def test_a_density_cannot_be_posted_at_all(client):
    """An evaluated density could contradict its claimed shape, so the save
    ignores extra keys."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _save(client, exp, hp,
                     {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                      "decay": {"shape": "none"},
                      "knots": [[0.0, -5.0], [1.0, 1.0]]})
    exp.refresh_from_db()

    assert response.status_code == 200
    assert "knots" not in exp.data.priors[hp]


def test_the_density_comes_back_with_the_save(client):
    """The reader's curve comes from the optimizer's own density function
    because the save returns the density."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _response, data = _slice(client, exp, hp)
    meta = data["meta"]

    body = {"hp": hp,
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
    # A Normal at 0.8 must peak at 0.8.
    peak = meta["positions"][grid.index(max(grid))]
    assert peak == pytest.approx(0.8, abs=2.0 / len(grid))


# The shadow beyond the slice


def test_the_payload_carries_a_sample_of_the_whole_space(client):
    """The shadow needs predictions that move the hyperparameters the slice
    keeps fixed."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    assert meta["cloud"] is not None, data.get("warning")
    cloud = meta["cloud"]
    assert len(cloud["positions"]) == len(cloud["mu"]) == len(cloud["sigma"])
    assert len(cloud["mu"]) > 100, "too few samples to estimate anything"


def test_the_server_sends_predictions_not_acquisition_values(client):
    """Expected improvement is only computed in the browser, so the server sends
    means and spreads."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    cloud = data["meta"]["cloud"]

    assert set(cloud) == {"positions", "mu", "sigma"}


def test_the_shadow_is_a_maximum_per_slot_not_a_mean():
    """The shadow plots each slot's maximum, the best the search could reach."""
    source = _script()

    assert "if (envelope[slot] === null || value > envelope[slot])" in source
    assert "function cloudSlots" in source, "slots do not depend on the prior"


def test_the_shadow_shares_the_curves_divisor():
    """A shared divisor makes the curves comparable and lets the shadow exceed
    1."""
    source = _script()

    assert "envelope[i] /= scale" in source


def test_the_shadow_is_named_for_what_it_is(client):
    """The shadow is a slack lower bound, and 25x the samples moved it from 0.17
    to 0.34 on one run."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    meta = data["meta"]

    traces = meta["traces"]["acquisition"]
    panel = _panel(data, "acquisition")["data"]
    cloud, ghost = panel[traces["cloud"]], panel[traces["cloudBare"]]

    # The names are translatable, so the test only checks that the pair differs.
    assert cloud["name"] and ghost["name"]
    assert cloud["name"] != ghost["name"], "the pair has to be tellable apart"


def test_the_controls_keep_their_space(client):
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
    """Without SMAC's acquisition weight layer the search ignores a saved prior,
    and only the warning tells the reader."""
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
    """A sigma cap below the axis width would block nearly flat priors."""
    source = _script()

    assert "min: 0.004, max: 1.0, step: 0.002" in source


# Prior decay

def _state(client, exp, hp, kind="normal", decay="logarithmic"):
    """`_state` posts through the endpoint, so `at_trial` is recorded as for a
    reader."""
    body = {"hp": hp,
            "prior": {"kind": kind, "params": {"mu": 0.6, "sigma": 0.2},
                      "decay": {"shape": decay}}}
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps(body), content_type="application/json")


def test_a_stated_prior_weakens_as_the_run_goes_on():
    """A decay factor taken from finished trials once made priors stronger, so
    the test asserts the direction."""
    # Only the pinned SMAC branch has a weight layer to decay.
    pytest.importorskip("smac.acquisition.weight",
                        reason="needs the SMAC branch carrying the acquisition weight layer")

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuilt_result(exp).trials)
    exp.data.priors = {hp: {"kind": "normal", "params": {}, "decay": "logarithmic",
                       "at_trial": at}}
    exp.data.save(update_fields=["priors"])

    seen = [_decayed(exp, hp, at + n)["exponent"] for n in (0, 1, 5, 20, 100)]

    assert seen == sorted(seen, reverse=True), seen
    assert seen[-1] < seen[0]


def test_a_prior_that_does_not_decay_keeps_its_strength():
    """A schedule that decayed anyway would still pass the test above."""

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuilt_result(exp).trials)
    exp.data.priors = {hp: {"kind": "normal", "params": {}, "decay": "none",
                       "at_trial": at}}
    exp.data.save(update_fields=["priors"])

    assert {_decayed(exp, hp, at + n)["exponent"] for n in (0, 5, 100)} == {1.0}


def test_the_figure_ships_the_exponent_that_currently_applies(client):
    """The server sends the exponent so SMAC's decay curves keep one definition."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    assert _state(client, exp, hp).status_code == 200

    _, data = _slice(client, exp, hp)

    assert isinstance(data["prior"]["exponent"], float)
    assert data["prior"]["decay"]["shape"] == "logarithmic"


# Next trials

def _smac_experiment():
    """Only a run with a surrogate and an acquisition function has anything to
    rank."""
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test.ihpo").read_bytes()))


def test_asking_the_optimizer_returns_a_ranked_slate(client):
    """`slice_challengers` replays the run's trials into its own facade and
    calls `ask`, so the candidates are what would run."""
    exp = _smac_experiment()
    hp = _hp_names(exp)[0]

    response = client.get(reverse("ui:next_trials", args=[exp.pk]) + f"?hp={hp}")
    candidates = json.loads(response.content)["candidates"]

    assert response.status_code == 200
    assert candidates
    # Display ranks: 1..n, no gaps, best first.
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    for c in candidates:
        assert 0.0 <= c["positions"][hp] <= 1.0
        assert c["origin"]
        # A candidate differs from the incumbent in every dimension.
        assert set(c["config"]) == set(_hp_names(exp))
        assert all(f"{k} =" in c["label"] for k in c["config"])


def test_saving_a_prior_asks_the_optimizer_nothing(client, monkeypatch):
    """Asking the optimizer costs a model fit and a maximization, so a save
    never asks.
    """
    from ui.views import priors

    asked = []
    monkeypatch.setattr(priors, "_next_trials", lambda *a, **k: asked.append(1))
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = _state(client, exp, hp)

    assert response.status_code == 200
    assert "candidates" not in json.loads(response.content)
    assert not asked


def test_an_optimizer_with_nothing_to_rank_says_so_rather_than_failing(client):
    """Random and grid search have nothing to rank, so the endpoint answers
    None."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    response = client.get(reverse("ui:next_trials", args=[exp.pk]) + f"?hp={hp}")

    assert response.status_code == 200
    assert json.loads(response.content)["candidates"] is None


# Initial design phase

def test_a_run_still_in_its_initial_design_says_so(client):
    """A prior cannot touch initial-design trials because the design runs before
    a surrogate exists."""
    from core.priors import initial_design_notice

    exp = _smac_experiment()
    result = _rebuilt_result(exp)

    class StillGoing:
        metadata = result.metadata
        trials = result.trials[:1]

    notice = initial_design_notice(StillGoing)

    assert notice is not None
    assert notice["done"] is False
    assert notice["trials"] == 1
    assert notice["size"] >= 1


def test_a_finished_design_raises_no_banner(client):
    from core.priors import initial_design_notice

    result = _rebuilt_result(_smac_experiment())

    class Finished:
        metadata = dict(result.metadata,
                        initial_design={"name": "SobolInitialDesign", "n_configs": 1,
                                        "additional_configs": []})
        trials = result.trials

    notice = initial_design_notice(Finished)
    assert notice is not None
    assert notice["done"] is True


def test_the_blackbox_fixture_never_left_its_initial_design(client):
    """The fixture asked for 32 initial points and recorded 30 trials, so a
    prior stated against it reaches nothing."""
    from core.priors import initial_design_notice

    notice = initial_design_notice(_rebuilt_result(_smac_experiment()))

    assert notice is not None
    assert notice["size"] == 32
    assert notice["trials"] == 30
    assert notice["done"] is False


def test_the_default_configuration_counts_toward_the_phase(client):
    """`use_default_config` adds the default to `additional_configs`, outside
    `n_configs` and inside the model-free phase."""
    from core.priors import initial_design_notice

    result = _rebuilt_result(_smac_experiment())

    class WithDefault:
        metadata = dict(result.metadata,
                        initial_design={"name": "SobolInitialDesign", "n_configs": 3,
                                        "additional_configs": [{"x": 1}]})
        trials = result.trials

    notice = initial_design_notice(WithDefault)
    assert notice is not None
    assert notice["size"] == 4


def test_the_size_is_read_and_never_guessed(client):
    """`initial_design_notice` reads the size and never counts origins, as
    `tests/core/test_initial_points.py` requires."""
    from core.priors import initial_design_notice

    result = _rebuilt_result(_smac_experiment())

    class Bare:
        metadata = {}
        trials = result.trials

    assert initial_design_notice(Bare) is None
    exp = _experiment()
    _, data = _slice(client, exp, _hp_names(exp)[0])
    assert data["initialDesign"] is None


def test_the_figure_carries_the_banner_element(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="acq-initial"' in html


# Decay factor β

def _state_beta(client, exp, hp, ratio, shape="quadratic"):
    body = {"hp": hp,
            "prior": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1},
                      "decay": {"shape": shape, "beta_ratio": ratio}}}
    return client.post(reverse("ui:save_prior", args=[exp.pk]),
                       data=json.dumps(body), content_type="application/json")


def test_beta_is_stored_as_a_ratio_of_the_budget(client):
    """πBO sets β to N/10 and DynaBO ablates N/50 to N/2.5, so β is stored as a
    budget fraction."""
    exp = _experiment()
    hp = _hp_names(exp)[0]

    assert _state_beta(client, exp, hp, 0.2).status_code == 200
    exp.refresh_from_db()

    assert exp.data.priors[hp]["decay"] == {"shape": "quadratic", "beta_ratio": 0.2}


def test_the_ratio_resolves_against_the_budget(client):
    from experiments.services.priors import run_budget

    exp = _experiment()
    hp = _hp_names(exp)[0]
    trials = len(_rebuilt_result(exp).trials)
    _state_beta(client, exp, hp, 0.2)
    exp.refresh_from_db()

    out = _decayed(exp, hp, trials)
    at = exp.data.priors[hp]["at_trial"]

    assert out["beta"] == pytest.approx(run_budget(exp, at) * 0.2)


def test_the_ratio_is_held_inside_the_ablated_range(client):
    """The bounds widen DynaBO's grid a little, and a ratio near one would keep
    the exponent above one."""
    from core.priors import BETA_RATIO_MAX, BETA_RATIO_MIN

    exp = _experiment()
    hp = _hp_names(exp)[0]
    for asked, expected in ((-5.0, BETA_RATIO_MIN), (100.0, BETA_RATIO_MAX)):
        _state_beta(client, exp, hp, asked)
        exp.refresh_from_db()
        assert exp.data.priors[hp]["decay"]["beta_ratio"] == expected


def test_an_absolute_beta_is_read_as_the_ratio_it_was(client):
    """Reading an absolute β as a ratio would make the prior about fifty times
    weaker."""
    from experiments.services.priors import run_budget

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuilt_result(exp).trials)
    budget = run_budget(exp, at)
    exp.data.priors = {hp: {"kind": "normal", "params": {}, "at_trial": at,
                       "decay": {"shape": "linear", "beta": budget * 0.2}}}
    exp.data.save(update_fields=["priors"])

    out = _decayed(exp, hp, at)

    assert out["decay"]["beta_ratio"] == pytest.approx(0.2)


def test_an_old_flat_decay_string_still_loads(client):

    exp = _experiment()
    hp = _hp_names(exp)[0]
    at = len(_rebuilt_result(exp).trials)
    exp.data.priors = {hp: {"kind": "normal", "params": {}, "decay": "linear",
                       "at_trial": at}}
    exp.data.save(update_fields=["priors"])

    out = _decayed(exp, hp, at + 5)

    assert out["decay"]["shape"] == "linear"
    assert out["decay"]["beta_ratio"] > 0
    assert out["exponent"] > 0


# Reading the panel

def test_the_sampled_envelope_keeps_an_unweighted_ghost(client):
    """The sampled envelope covers the whole space, so its unweighted twin shows
    what the reader's prior changed.
    """
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    indices = data["meta"]["traces"]["acquisition"]
    panel = _panel(data, "acquisition")["data"]

    assert panel[indices["cloudBare"]]["line"]["dash"] == "dot"
    assert panel[indices["cloud"]]["line"].get("dash") is None
    # Behind the weighted curve, the one being read.
    assert indices["cloudBare"] < indices["cloud"]


def test_the_ghost_is_empty_without_a_prior(client):
    source = _script()

    assert "bare = stated && cloudWeight ? [] : null;" in source
    assert "values.cloudBare || []" in source


def test_the_candidate_hover_is_anchored_to_its_marker(client):
    """Every curve here skips hover, so the layout anchors each label to its
    candidate marker, which Plotly keeps in view."""
    exp = _experiment()
    _response, data = _slice(client, exp, _hp_names(exp)[0])
    layout = _panel(data, "acquisition")["layout"]

    assert layout["hovermode"] == "closest"
    assert layout["hoverlabel"]["align"] == "left"
    for name in ("acquisition", "weighted"):
        index = data["meta"]["traces"]["acquisition"][name]
        assert _panel(data, "acquisition")["data"][index]["hoverinfo"] == "skip"


def test_a_pile_of_candidates_does_not_spell_every_one_out(client):
    """A cluster spells out only its best member, and the template contains the
    wording for translators."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "candidates here" in html
    assert "most promising" in html
    assert "HOVER_MEMBERS" not in _script()


def test_the_script_writes_no_english_of_its_own():
    """Strings in acquisition.js reach no translator, so the script reads them
    from `acq-strings`."""
    import re

    source = _script()
    stripped = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    stripped = re.sub(r'text\("[a-z-]+"', "", stripped)
    leftover = [m for m in re.findall(r'"([A-Z][a-z]+(?: [a-z]+){2,}[^"]*)"', stripped)]

    assert not leftover, leftover


def test_the_prior_panel_leaves_room_above_a_full_strength_belief(client):
    """Against the top edge, the flat line of a uniform prior looked like the
    frame. The browser converts pointer positions against the same axis top, so
    the test checks both numbers.
    """
    from ui.figures.plots import PRIOR_AXIS_TOP

    exp = _experiment()
    _, data = _slice(client, exp, _hp_names(exp)[0])
    axis = _panel(data, "prior")["layout"]["yaxis"]

    assert axis["range"] == [0, PRIOR_AXIS_TOP]
    assert PRIOR_AXIS_TOP >= 1.2, "a flat line at 1 has to sit clear of the edge"
    assert f"PRIOR_AXIS_TOP = {PRIOR_AXIS_TOP}" in _script(), (
        "the browser converts pointer positions against this; it cannot drift")


def test_a_prior_only_payload_describes_figures_it_does_not_carry(client):
    """`meta` describes all three panels while a prior-only payload only carries
    the prior figure, so `meta.traces.acquisition.candidates` exists with no
    acquisition figure behind it.
    """
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": hp, "prior_only": "1"})
    data = json.loads(response.content)

    assert set(data["figures"]) == {"prior"}
    assert "candidates" in data["meta"]["traces"]["acquisition"]


def test_the_candidate_trace_is_guarded_on_the_figure_not_the_index():
    """`setCandidates` runs on prior-only payloads too, where reading
    `state.figures.acquisition.data` after guarding on `meta.traces` threw
    inside a silent `.catch`.
    """
    source = _script()

    assert "|| !state.figures || !state.figures.acquisition) return false;" in source


def test_a_failed_prior_draw_is_not_swallowed():
    """An exception while drawing means broken code, so the catch logs it."""
    source = _script()
    catch = source[source.index("function drawPriorOnly"):]
    catch = catch[:catch.index("delete inflight[key]")]

    assert "console.error" in catch, "a broken draw has to say so somewhere"
