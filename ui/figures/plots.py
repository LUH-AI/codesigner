"""Plotly chart builders, one per figure that draws a figure.

Each is named for the figure it backs (see `catalog.py`), so
`performance_over_time_plot` fills the "Performance over time" figure. The
other figures — best/selected configuration and trials — are tables, and are
built by their templates from the view's context rather than from here; so is
hyperparameter importance's "table" view, alongside this module's pie/bar.

Pure functions: given an OptimizationResult and a metric, return a
plotly.graph_objects.Figure (or None). No Django, no request state — the view
serializes these with fig.to_json() and the browser renders them.

Every per-metric builder returns None rather than drawing a partial picture
when the result has no score for that metric on some trial — see
`OptimizationResult.has_every_score` for why that is a whole-figure decision
rather than a per-point one.
"""

import math

import plotly.graph_objects as go

from core.projection import log_hyperparameters, project

from ..formatting import sigfigs

# ── The palette ──────────────────────────────────────────────────────────────
#
# Four colours across every figure, each meaning one thing. Held by convention
# rather than by machinery — nothing checks it — but a reader who learns a
# colour on one figure should not have to unlearn it on the next.
#
#   MARKER_COLOR      a trial, an observation, an unemphasised line; and the
#                     positive side of a signed quantity (synergy, "helped")
#   SELECTION_COLOR   the selected trial, and nothing else
#   ACCENT_COLOR      the best-so-far: the incumbent line, the trials that
#                     improved it, and the partial-dependence mean, which is
#                     the same idea one level up — what the surrogate makes of
#                     the run rather than one point in it
#   NEGATIVE_COLOR    the negative side of a signed quantity: redundancy
#                     between hyperparameters, a value that hurt
#   _INTENSITY_SCALE  anything scaled to the metric, on every figure that
#                     scales anything to the metric (see below)
#
# The first two are public because the page draws the selection highlight
# itself — a restyle, not a rebuilt figure — see views.py's `selection_colors`.
MARKER_COLOR = "#636EFA"
SELECTION_COLOR = "#EF553B"
ACCENT_COLOR = "#10B981"
NEGATIVE_COLOR = "#FF2B2B"


#: How a figure is able to show which trial is selected. Not a style choice —
#: it is decided by what the figure's colour already means.
RECOLOR = "recolor"   # marker colour is free: paint the selected point with it
OUTLINE = "outline"   # colour already carries the score or the sign: ring it
LINE = "line"         # a trial is a polyline, not a point (parallel coordinates)
OVERLAY = "overlay"   # the selected point again, on top, in the selection colour


def _signed(value) -> str:
    """A signed quantity as a reader sees it: the sign, then four figures.

    The sign is the reading on these figures — helped or hurt, synergy or
    redundancy — so it is written even when it is positive, which `sigfigs`
    alone does not do.
    """
    return f"{'+' if value >= 0 else '-'}{sigfigs(abs(value))}"


def _selection_meta(by_trace: dict, *, style, segment=1, **extra) -> dict:
    """Where a plot's per-trial points are, for the page's selection bus.

    One trial is selected at a time across the whole page, so every figure that
    draws all the trials has to be able to say two things: which trial a clicked
    point belongs to, and which points to light up when the selection changes
    elsewhere. Both come from here, in `layout.meta["selection"]`, so the script
    needs no per-figure knowledge — see experiment_detail.html.

    *by_trace* maps a trace index to the positions in `result.trials` its points
    stand for, in point order. Keyed by trace because a trace not named here is
    not trials at all — an incumbent line, a colour-bar carrier — and a click on
    one means nothing. Positions rather than `TrialResult.trial` numbers because
    that is what the trial-panel endpoint and the local-ablation cache already
    key on, and named rather than implied by position because local effects
    draws a *sample*: its nth point is not its nth trial.

    *segment* is how many points make up one trial, which is 1 everywhere except
    parallel coordinates, where a trial is a whole polyline. So one rule reads
    every figure: the trial at point *p* of trace *t* is
    `by_trace[t][p // segment]`.

    *style* is RECOLOR / OUTLINE / LINE, above.
    """
    return dict(trials={str(k): list(v) for k, v in by_trace.items()},
                style=style, segment=segment, **extra)


def incumbent_scores(result, display_metric):
    """The running best (non-decreasing) score by *display_metric*, per trial.

    Now a thin delegate: the computation lives on `OptimizationResult`, which
    memoizes it per metric so the four `performance_over_time` views stop
    recomputing the identical list. Kept as a function because it is part of
    `ui.figures`' exported surface.
    """
    return result.incumbent_scores(display_metric)


def performance_over_time_plot(result, display_metric, *, x_axis="trial",
                                y_axis="score", selected_idx=None):
    """Every trial's outcome plus the running-best line, on whichever axes
    *x_axis* ("trial" | "time") and *y_axis* ("score" | "error") pick — the
    four views of one underlying curve, replacing what used to be two
    separately-drawn figures (performance-over-trials, error-over-time).

    The x-axis is trial index or cumulative trial duration — compute *spent*,
    not wall-clock elapsed since the experiment was created, so a gap between
    resumed runs (which can be arbitrarily long once experiments persist)
    never shows up as a dead stretch on the time axis. y is the raw score or
    1-minus-it floored at 1e-3 (so log scale never hits zero); either way the
    incumbent line is the running best in that same unit, drawn as a step
    (`shape="hv"`) since it only actually changes at an improvement, plus
    markers flagging exactly which trials those were.

    The point at *selected_idx* (default: none) is enlarged and recolored on
    the raw-outcome trace only — the same highlight click-to-select drives.
    Returns None with no trials.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None

    if x_axis == "time":
        xs, running = [], 0.0
        for t in trials:
            running += t.duration
            xs.append(running)
        x_title = "Time (s)"
    else:
        xs = [t.trial for t in trials]
        x_title = "Trial"

    incumbents = incumbent_scores(result, display_metric)
    if y_axis == "error":
        ys = [max(1e-3, 1.0 - t.scores[display_metric]) for t in trials]
        incumbent_ys = [max(1e-3, 1.0 - v) for v in incumbents]
        y_title, y_type, outcome_name = "Error", "log", "Trial error"
    else:
        ys = [t.scores[display_metric] for t in trials]
        incumbent_ys = incumbents
        y_title, y_type, outcome_name = display_metric.capitalize(), "linear", "Trial score"

    colors = [MARKER_COLOR] * len(trials)
    sizes = [6] * len(trials)
    if selected_idx is not None and 0 <= selected_idx < len(trials):
        colors[selected_idx] = SELECTION_COLOR
        sizes[selected_idx] = 13

    improved = [i == 0 or incumbents[i] > incumbents[i - 1] for i in range(len(trials))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=outcome_name,
        marker=dict(size=sizes, color=colors, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=incumbent_ys, mode="lines", name="Incumbent",
        line=dict(width=2, shape="hv", color=ACCENT_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=[xs[i] for i in range(len(trials)) if improved[i]],
        y=[incumbent_ys[i] for i in range(len(trials)) if improved[i]],
        mode="markers", name="New incumbent",
        marker=dict(size=9, color=ACCENT_COLOR, symbol="diamond"),
        # Out of the way of the pointer. These are drawn over the trials they
        # mark and are bigger than them, so with hover on they sat between the
        # reader and exactly the trials most worth selecting — and they are not
        # trials, so a click on one names nothing. Skipping hover takes them out
        # of Plotly's hit-testing entirely and the point beneath answers instead;
        # nothing is lost, since it is the same trial and it says so.
        hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=y_title, yaxis_type=y_type,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
        # Trace 0 only: the incumbent line and the new-incumbent diamonds are
        # drawn from the same trials but are not a trial each, and a click on
        # one names nothing.
        meta={"selection": _selection_meta({0: range(len(trials))},
                                           style=RECOLOR)},
    )
    return fig


def hyperparameter_importance_plot(importance: dict, rendering: str = "pie",
                                   title: str = "Importance"):
    """Pie or (vertical) bar of a per-hyperparameter importance dict — shared
    by every HyperSHAP game this app surfaces (tunability, sensitivity,
    mistunability all produce the same shape: {hp: normalized weight}), the
    caller picks which dict to hand in. A third rendering, "table", is
    rendered directly by the template from `panels`, not from here.

    How much of each hyperparameter's value has already been captured is
    `hyperparameter_progress_plot`, which draws the same two renderings split.

    Returns None when *importance* is empty, or when *rendering* is "table"
    (nothing to draw), so the caller shows the explanatory message / the
    table instead.
    """
    if not importance or rendering == "table":
        return None
    if rendering == "bar":
        # Descending, left to right — DeepCAVE's own reading order for this,
        # and it suits a vertical bar better than the ascending order a
        # horizontal one wants (largest nearest the axis label).
        names, values = zip(*sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=MARKER_COLOR))
        fig.update_layout(yaxis_title=title,
                          margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        return fig
    fig = go.Figure(go.Pie(
        labels=list(importance.keys()), values=list(importance.values()),
        textinfo="label+percent", textposition="inside",
        insidetextorientation="horizontal", textfont=_TEXT_FONT,
        # Stated rather than left out. `Plotly.react` diffs against what is
        # already drawn, and an attribute that is simply absent is a weaker
        # instruction than one set to its default.
        hole=0,
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
                      **_UNIFORM_TEXT)
    return fig


#: One text size for every wedge, and no text at all where it will not fit.
#:
#: Two settings, because it takes both. `textfont` fixes the size — Plotly
#: otherwise scales a label to its own slice, so a thin wedge gets tiny text and
#: a fat one gets large text, which reads as emphasis that is not there and is
#: unreadable at the small end. `uniformtext` then decides what happens to a
#: label that no longer fits at that size: `mode: "hide"` drops it rather than
#: shrinking it back, and the hover still carries it.
#:
#: `minsize` matches the font size on purpose. Below it, Plotly would be
#: scaling again, which is the thing being prevented.
_TEXT_SIZE = 12
_TEXT_FONT = {"size": _TEXT_SIZE}
_UNIFORM_TEXT = {"uniformtext": {"minsize": _TEXT_SIZE, "mode": "hide"}}

#: The two parts of a hyperparameter's achievable gain, darkest first. One hue
#: at two intensities, like `hyperparameter_orders_plot`'s three: these are parts
#: of one quantity rather than two quantities, and a second hue would say they
#: were different things. Darker is what is still to gain, because that is the
#: part worth acting on.
_PROGRESS_SHADES = (MARKER_COLOR, "#9DA5F5")


def hyperparameter_progress_plot(rows: list, rendering: str = "pie"):
    """Each hyperparameter's achievable gain, split into banked and still to gain.

    The same two renderings the importance figure offers, of the same
    quantity — but each slice or bar divided, so "this one is worth a lot and we
    have already taken almost all of it" is one shape rather than two figures
    compared by eye. `rows` is `_tuning_progress`'s output: name, achievable,
    banked, remaining.

    **bar** stacks banked under still-to-gain, so a bar that is nearly all light
    is a settled question and one that is nearly all dark is where the budget
    should go. **pie** is a sunburst: the inner ring is the hyperparameters,
    sized as the plain pie sizes them, and the outer ring divides each wedge into
    the same two parts.

    A sunburst rather than a second ring whose *radius* encodes the fraction —
    that would ask the reader to compare arcs at different radii, which reads
    badly and reads worse the more hyperparameters there are. Here the split is
    an angle inside its own wedge, which is the same comparison the pie already
    asks for.

    Banked is clamped into 0..achievable for drawing: the ablation can exceed
    the max game's estimate or go negative, since they are two estimates of the
    same surface, and neither is a length. The table keeps the true signed value.

    Returns None with no rows, or when *rendering* is "table" (the table shows
    the numbers themselves).
    """
    if not rows or rendering == "table":
        return None

    ordered = sorted(rows, key=lambda row: row["achievable"], reverse=True)
    names = [row["name"] for row in ordered]
    banked = [min(max(0.0, row["banked"]), row["achievable"]) for row in ordered]
    left = [row["achievable"] - value for row, value in zip(ordered, banked)]
    if not any(banked) and not any(left):
        return None

    if rendering == "bar":
        fig = go.Figure()
        # Banked first, so it is the base of the bar and the remainder sits on
        # top of it — a bar filling up rather than emptying out.
        for values, label, shade in ((banked, "Already banked", _PROGRESS_SHADES[1]),
                                     (left, "Still to gain", _PROGRESS_SHADES[0])):
            fig.add_trace(go.Bar(x=names, y=values, name=label, marker_color=shade,
                                 hovertemplate="%{x}<br>" + label
                                               + " %{y:.4g}<extra></extra>"))
        fig.update_layout(
            barmode="stack", yaxis_title="Achievable gain",
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1))
        return fig

    # One flat pie with two slices per hyperparameter, adjacent — so a wedge
    # is literally part light and part dark, which is the thing being shown.
    #
    # This was a sunburst first. A sunburst puts the split on an outer ring,
    # which is legible enough, but with several hyperparameters there is no
    # single root node to fill the centre and it leaves a hole there; the labels
    # then sit on the innermost ring, where there is least room for them. A flat
    # pie has neither problem and asks for one less idea from the reader.
    labels, values, colours, hovers, captions = [], [], [], [], []
    for name, taken, remaining in zip(names, banked, left):
        for part, value, shade in (("banked", taken, _PROGRESS_SHADES[1]),
                                   ("still to gain", remaining, _PROGRESS_SHADES[0])):
            labels.append(f"{name} — {part}")
            values.append(value)
            colours.append(shade)
            hovers.append(f"{name}<br>{part} {sigfigs(value)}")
        # Named once per hyperparameter, on whichever of its two parts has the
        # room. Both labelled would say it twice; the smaller one labelled would
        # often be a label with nowhere to go.
        captions.extend([name, ""] if taken >= remaining else ["", name])

    fig = go.Figure(go.Pie(
        labels=labels, values=values, text=captions, textinfo="text",
        textposition="inside", insidetextorientation="horizontal",
        textfont=_TEXT_FONT, sort=False, direction="clockwise", hole=0,
        marker=dict(colors=colours, line=dict(width=1, color="#FFFFFF")),
        hovertext=hovers, hoverinfo="text",
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
                      **_UNIFORM_TEXT)
    return fig


def hyperparameter_ablation_plot(ablation: dict):
    """Waterfall of one trial's HyperSHAP ablation values against the config
    space's default — the "Local (selected trial)" view.

    Reads as a walk: start at the default configuration, change one
    hyperparameter to what this trial used, see where the score goes, repeat.
    The running total is the point — a diverging bar (which this replaces)
    shows the same numbers but leaves the reader to add them up, and the sum is
    the thing they actually want, namely how far this trial got from the
    default and which changes carried it.

    Signed, deliberately: a step up means that hyperparameter's value in this
    trial beat the default, down means it lost to it. Ordered by magnitude, so
    the steps that decided the outcome come first rather than being buried in
    the middle of the walk.

    The axis is *relative to the default*, not an absolute score: the ablation
    game answers "how much did each choice help", and the default's own score
    is not among the numbers it returns. Both ends are labelled so that reads
    plainly rather than looking like a score that starts at zero.

    Returns None when *ablation* is empty (not enough trials, or the game
    failed — see BaseOptimizer.compute_hp_ablation). No "table"/"pie" — a
    share-of-a-whole framing does not apply to signed values, so this is the
    one view this game gets.
    """
    if not ablation:
        return None
    names, values = zip(*sorted(ablation.items(), key=lambda kv: abs(kv[1]), reverse=True))
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(names) + ["total"],
        x=["Default"] + list(names) + ["This trial"],
        y=[0.0] + list(values) + [0.0],
        text=[""] + [_signed(v) for v in values] + [""],
        textposition="outside",
        increasing=dict(marker=dict(color=MARKER_COLOR)),
        decreasing=dict(marker=dict(color=NEGATIVE_COLOR)),
        totals=dict(marker=dict(color="#8C9196")),
        connector=dict(line=dict(color="rgba(140,140,140,0.5)", width=1)),
    ))
    fig.update_layout(yaxis_title="vs. default",
                      margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
    return fig


def hyperparameter_interactions_heatmap_plot(interactions: dict):
    """Hyperparameter x hyperparameter heatmap of pairwise (order-2) HyperSHAP
    tunability interactions — the "Hyperparameter interactions" figure's
    default view. The diagonal is that hyperparameter's own signed order-1
    value; off-diagonal cells are the signed pairwise interaction between the
    two — positive reads as synergy (the pair matters more tuned together
    than the sum of tuning each alone), negative as redundancy.

    Hyperparameters are ordered by |diagonal value| descending, the same
    "most important first" convention the bar views use elsewhere, so the
    strongest single effects sit nearest the top-left corner.

    Returns None when *interactions* is empty (not enough trials, or the
    game failed — see BaseOptimizer.compute_hp_interactions).
    """
    if not interactions:
        return None
    params = sorted(interactions.keys(), key=lambda p: abs(interactions[p][p]), reverse=True)
    z = [[interactions[a][b] for b in params] for a in params]
    fig = go.Figure(go.Heatmap(
        z=z, x=params, y=params, colorscale=_SIGN_SCALE, zmid=0,
        colorbar=dict(title="Interaction"),
    ))
    fig.update_layout(margin=dict(t=20, b=40, l=80, r=20))
    return fig


def hyperparameter_interactions_bar_plot(interactions: dict, top_k: int = 10):
    """Bar of the *top_k* strongest off-diagonal pairwise interactions, ranked
    by magnitude — an alternate view of the heatmap's same data, for reading
    off the handful of pairs that matter without scanning a grid.

    Signed and colored by sign, like `hyperparameter_ablation_plot` — positive
    is synergy, negative is redundancy, and which one loses meaning if
    flattened to a magnitude.

    Returns None when *interactions* is empty, or there are fewer than two
    hyperparameters (no pairs to show).
    """
    if not interactions:
        return None
    params = list(interactions.keys())
    pairs = [(f"{a} × {b}", interactions[a][b])
             for i, a in enumerate(params) for b in params[i + 1:]]
    if not pairs:
        return None
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    names, values = zip(*pairs[:top_k])
    colors = [MARKER_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    fig.update_layout(yaxis_title="Interaction strength",
                      margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    return fig


#: Score-carrying figures (the cube, parallel coordinates) ramp one hue by
#: intensity rather than travelling between two. A two-ended scale invites the
#: reader to find a meaningful middle, and there isn't one — a score has a good
#: end and a bad end and nothing special in between, so "more colour is better"
#: is the whole legend. Light-to-dark in the same blue the rest of the app
#: already uses for a plain data mark, so the figures read as one family.
_INTENSITY_SCALE = [
    [0.00, "#EEF1FE"],
    [0.25, "#C3CBFA"],
    [0.50, "#93A0F4"],
    [0.75, "#6373EC"],
    [1.00, "#2B3AA8"],
]

#: Positive interactions read as synergy, negative as redundancy — the same
#: two colours the ablation and top-pairs views already use for sign.
_SYNERGY_COLOR = MARKER_COLOR
_REDUNDANCY_COLOR = NEGATIVE_COLOR

#: The same two, as a diverging scale through white, for the figure that shows
#: sign as a fill rather than as a line.
#:
#: Built here rather than asking for Plotly's named "RdBu", which is two
#: different scales depending on who resolves it: the bundled plotly.min.js runs
#: it blue at 0 to red at 1, so with zmid=0 it put *positive* on red — the
#: reverse of every other signed figure in this app — with an orange band around
#: 0.6-0.7 belonging to no meaning at all. (Python's own scale of that name runs
#: the other way, which is why reading the two sides disagrees.)
_SIGN_SCALE = [
    [0.0, NEGATIVE_COLOR],
    [0.5, "#F7F7F7"],
    [1.0, MARKER_COLOR],
]


def _circle_positions(names: list) -> dict:
    """Each name at an even angle on the unit circle, first at twelve o'clock.

    The whole layout. HyperSHAP draws its own SI graph this way — it passes
    `nx.circular_layout` with `compactness=1e50`, which pins the nodes to the
    circle and leaves the force-directed step nothing to do — so reproducing it
    needs no graph library and no layout algorithm, just cos and sin. That is
    what makes this figure Plotly-native rather than a static image, and so the
    only one in the app that would have broken the "every figure is live"
    property is not.
    """
    import math

    n = len(names)
    return {name: (math.sin(2 * math.pi * i / n), math.cos(2 * math.pi * i / n))
            for i, name in enumerate(names)}


def hyperparameter_graph_plot(moebius: list, top_k: int = 15):
    """The Möbius interaction graph: hyperparameters on a circle, interactions
    between them as edges.

    Each Möbius term of size 2 is an edge between its two hyperparameters. Each
    term of size 3 or more is a *hyper*-edge — a small hub at the members'
    centroid with a spoke to each of them, which is how shapiq draws the same
    thing and the only honest way to show that four hyperparameters interact
    *as a set* rather than as six pairs. Order-1 terms size the nodes.

    Width is the term's magnitude; colour is its sign — synergy where a
    coalition is worth more than its parts, redundancy where it is worth less.

    This is the only view in the app that shows anything above order 2, and the
    reason the Möbius decomposition is stored at all: FSII's terms shift when
    the truncation order changes, so they cannot be read at several orders at
    once (see `BaseOptimizer._shared_exact_computer`).

    Only the *top_k* strongest interactions are drawn. A 6-hyperparameter run
    has 57 of them and a 10-hyperparameter one has 1,013; past a dozen or so the
    picture stops being a graph and becomes a ball of wool. shapiq caps its own
    for the same reason.

    Returns None when *moebius* is empty, or names fewer than two
    hyperparameters (nothing can interact with itself).
    """
    if not moebius:
        return None

    names = sorted({m for row in moebius for m in row["members"]})
    if len(names) < 2:
        return None
    pos = _circle_positions(names)
    own = {row["members"][0]: abs(row["value"])
           for row in moebius if len(row["members"]) == 1}
    biggest_own = max(own.values(), default=0.0) or 1.0

    # Sorted here rather than trusted from the caller: the stored list is
    # already ranked (see `_moebius_terms`), but a top-k slice that silently
    # depends on someone else's ordering is the kind of coupling that breaks
    # quietly when a second caller appears.
    edges = sorted((row for row in moebius if len(row["members"]) > 1),
                   key=lambda row: abs(row["value"]), reverse=True)[:top_k]
    widest = max((abs(row["value"]) for row in edges), default=0.0) or 1.0

    shapes, hub_x, hub_y, hub_text = [], [], [], []
    for row in edges:
        members, value = row["members"], row["value"]
        width = 1.0 + 7.0 * abs(value) / widest
        colour = _SYNERGY_COLOR if value >= 0 else _REDUNDANCY_COLOR
        label = " × ".join(members) + f"<br>{_signed(value)}"
        if len(members) == 2:
            (x0, y0), (x1, y1) = pos[members[0]], pos[members[1]]
            shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                               line=dict(color=colour, width=width), opacity=0.6, layer="below"))
            hub_x.append((x0 + x1) / 2); hub_y.append((y0 + y1) / 2)
        else:
            cx = sum(pos[m][0] for m in members) / len(members)
            cy = sum(pos[m][1] for m in members) / len(members)
            for member in members:
                mx, my = pos[member]
                shapes.append(dict(type="line", x0=mx, y0=my, x1=cx, y1=cy,
                                   line=dict(color=colour, width=width), opacity=0.5,
                                   layer="below"))
            hub_x.append(cx); hub_y.append(cy)
        hub_text.append(label)

    fig = go.Figure()
    # The hubs carry the hover text for every interaction, because a `shape` has
    # no hover of its own — without this the picture would be unreadable in
    # exactly the cases it exists for, where several edges cross.
    fig.add_trace(go.Scatter(
        x=hub_x, y=hub_y, mode="markers", name="Interaction",
        marker=dict(size=9, color="rgba(0,0,0,0)",
                    line=dict(width=1, color="rgba(120,120,120,0.5)")),
        text=hub_text, hoverinfo="text", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names],
        mode="markers+text", name="Hyperparameter",
        marker=dict(size=[14 + 26 * own.get(n, 0.0) / biggest_own for n in names],
                    color=MARKER_COLOR, opacity=0.85),
        text=names, textposition="top center",
        hovertext=[f"{n}<br>on its own {sigfigs(own.get(n, 0.0))}" for n in names],
        hoverinfo="text", showlegend=False,
    ))
    # No `scaleanchor`: locking the axes to a 1:1 ratio makes Plotly letterbox
    # the circle inside whichever dimension is scarcer, which on a card that is
    # wider than it is tall leaves the graph a small disc floating in white
    # space. The circle is a layout convention, not a measurement — nothing is
    # read off the distance between two nodes — so letting it stretch to an
    # ellipse that fills the card loses nothing and gains all the room the
    # labels need. The x range is wider than the y to leave that room, since
    # names sit beside the leftmost and rightmost nodes rather than above them.
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.25, 1.3]),
        margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
    )
    return fig


def hyperparameter_upset_plot(moebius: list, top_k: int = 12):
    """UpSet plot of the strongest Möbius coalitions: a ranked bar per
    coalition over a matrix saying which hyperparameters are in it.

    The same data as the graph, read the other way round. The graph is spatial
    and shows structure at a glance; it also stops being legible somewhere
    around six hyperparameters, which is exactly where a ranked list starts
    earning its place. So they are two views of one thing rather than a choice
    of one — a wide search wants this, a narrow one wants the graph.

    Filled dots are the members of a coalition, joined by a rule; hollow dots
    are the hyperparameters left out. Bars are signed, so a redundant coalition
    hangs below the axis rather than being flattened into a magnitude.

    Returns None when *moebius* holds no coalition of two or more (a run with a
    single hyperparameter, or one that fell back to the surrogate).
    """
    from plotly.subplots import make_subplots

    coalitions = sorted((row for row in moebius or [] if len(row["members"]) > 1),
                        key=lambda row: abs(row["value"]), reverse=True)[:top_k]
    if not coalitions:
        return None

    names = sorted({m for row in moebius for m in row["members"]})
    columns = [" × ".join(row["members"]) for row in coalitions]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)
    fig.add_trace(go.Bar(
        x=columns, y=[row["value"] for row in coalitions],
        marker_color=[_SYNERGY_COLOR if row["value"] >= 0 else _REDUNDANCY_COLOR
                      for row in coalitions],
        hovertemplate="%{x}<br>%{y:+.4g}<extra></extra>", showlegend=False,
    ), row=1, col=1)

    # Absent members first so the filled ones draw over them, and one rule per
    # column joining its extremes — without it a coalition spread across
    # non-adjacent rows reads as unrelated dots.
    out_x, out_y, in_x, in_y = [], [], [], []
    for column, row in zip(columns, coalitions):
        for name in names:
            (in_x if name in row["members"] else out_x).append(column)
            (in_y if name in row["members"] else out_y).append(name)
        members = [n for n in names if n in row["members"]]
        fig.add_trace(go.Scatter(
            x=[column, column], y=[members[0], members[-1]], mode="lines",
            line=dict(color=MARKER_COLOR, width=2), hoverinfo="skip", showlegend=False,
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=out_x, y=out_y, mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=9, color="rgba(0,0,0,0)",
                    line=dict(width=1, color="rgba(140,140,140,0.45)")),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=in_x, y=in_y, mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=10, color=MARKER_COLOR),
    ), row=2, col=1)

    fig.update_yaxes(title_text="Interaction", row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=names, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(margin=dict(t=20, b=20, l=90, r=20), showlegend=False)
    return fig


#: How Möbius terms are bucketed for the by-order view. Order 3 and above share
#: a band: past pairs the distinction stops being actionable ("this only matters
#: in a group" reads the same at 3 as at 5) and a band per order would give a
#: 10-hyperparameter run ten of them.
_ORDER_BANDS = (
    (1, 1, "On its own"),
    (2, 2, "In pairs"),
    (3, None, "In larger groups"),
)


def hyperparameter_orders_plot(moebius: list):
    """Stacked bar of where each hyperparameter's influence comes from: itself,
    its pairings, or larger groups — the "By order" view of the interactions
    figure.

    Answers a question none of the other views can: *does this hyperparameter
    matter on its own, or only in combination?* A hyperparameter with a tall
    solid first band is worth tuning by itself; one that is nearly all upper
    bands only pays off alongside something else, which is a different piece of
    advice.

    Reads the Möbius decomposition rather than the order-2 FSII grid, because
    that is the one that assigns a single value per coalition of any size — see
    `BaseOptimizer._shared_exact_computer` for why the two are not
    interchangeable.

    Each term is split equally among its members, which is the standard reading
    of a Harsanyi dividend: a value that belongs to the coalition as a whole and
    to no member in particular. Magnitudes, not signed values — a stack of
    signed contributions would cancel and show a hyperparameter that matters a
    great deal in both directions as one that matters not at all.

    Bars are ordered by total height, so the same "most important first"
    convention the other views use. Returns None when *moebius* is empty (a run
    from before this was stored, or a game that fell back).
    """
    if not moebius:
        return None

    totals: dict = {}
    for row in moebius:
        members, share = row["members"], abs(row["value"]) / len(row["members"])
        for band_index, (low, high, _label) in enumerate(_ORDER_BANDS):
            if len(members) >= low and (high is None or len(members) <= high):
                for member in members:
                    totals.setdefault(member, [0.0] * len(_ORDER_BANDS))[band_index] += share
                break

    if not totals:
        return None
    names = sorted(totals, key=lambda n: sum(totals[n]), reverse=True)
    shades = [MARKER_COLOR, "#9DA5F5", "#D3D6FB"]

    fig = go.Figure()
    for band_index, (_low, _high, label) in enumerate(_ORDER_BANDS):
        fig.add_trace(go.Bar(
            x=names, y=[totals[n][band_index] for n in names],
            name=label, marker_color=shades[band_index],
        ))
    fig.update_layout(
        barmode="stack", yaxis_title="Attributed influence",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def configuration_cube_plot(result, display_metric, config_space=None):
    """Every trial as one point in hyperparameter space, colored by its
    *display_metric* score — DeepCave's "Configuration Cube." Like DeepCave's,
    the axes are two (or, client-side, three) actual hyperparameters rather
    than a projection, so a position on an axis is a literal hyperparameter
    value. (The MDS-projected view is DeepCave's separate *Footprint* plugin,
    `deepcave/evaluators/footprint.py`; this is not a version of that.) What
    does differ is the picker: dropdowns here, a checklist plus a
    "how many configurations" slider there.

    Nothing is on an axis until someone puts it there: all three pickers start
    at None, and how many of them are filled is what decides whether this draws
    nothing, a strip, a plane or a cube. Which hyperparameter belongs on which
    axis is the question the figure exists to ask, and a server-chosen pair
    answers it before it is asked — the first two in config-space order are
    only "first" in the sense that something had to be.

    So this ships the data without an axis assignment: every hyperparameter's
    full per-trial values ride along on the trace's `customdata` (one row per
    trial, columns in `layout.meta["hp_names"]` order) with `x`/`y` left empty,
    and the client fills them from whichever columns are picked — no server
    round trip per axis change (see experiment_detail.html's applyCubeAxes).
    Self-contained in this one trace's own payload on purpose, so it works
    regardless of whether the Trials figure (which also lists every
    hyperparameter) is toggled on.

    A hyperparameter the search treats logarithmically gets a logarithmic
    axis, and since the axes are assigned in the browser the names go in
    `layout.meta["log_hps"]` for `applyCubeAxes` to read. Without it, `C` on
    the SVM model — sampled log-uniformly over 0.01 to 100 — puts three
    quarters of its points in the bottom tenth of a linear axis, which hides
    exactly the region the search spent its time in. DeepCAVE solves the same
    problem by plotting the normalized value and relabelling the ticks; a log
    axis keeps the tick values real numbers, which reads better and costs
    nothing here.

    Returns None with no trials or no hyperparameters. One is enough to draw:
    a single axis is a real answer to "where did the search go", and refusing
    it would leave a model with one hyperparameter with no cube at all.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None
    hp_names = list(trials[0].config.keys())
    if not hp_names:
        return None
    customdata = [[t.config.get(h) for h in hp_names] for t in trials]
    scores = [t.scores[display_metric] for t in trials]
    fig = go.Figure(go.Scatter(
        x=[], y=[], mode="markers", customdata=customdata,
        text=[f"Trial {t.trial}" for t in trials],
        marker=dict(size=8, color=scores, colorscale=_INTENSITY_SCALE, showscale=True,
                    colorbar=dict(title=display_metric.capitalize()),
                    # Opaque, and each point ringed in the card's own colour.
                    # Left to blend, two points that nearly overlap composite
                    # into something darker than either — which on a scale where
                    # darker means better is a trial that did not happen. The
                    # ring makes an overlap read as two points rather than as
                    # one good one. (`opacity` defaults to 1; said out loud
                    # because it is load-bearing here rather than incidental.)
                    opacity=1, line=dict(width=1, color="#FFFFFF")),
    ))
    fig.update_layout(
        meta={"hp_names": hp_names,
              "log_hps": log_hyperparameters(config_space, hp_names),
              # The selected trial drawn again on top, in the selection
              # colour, rather than recoloured in place: the fill is the score
              # here, so painting one point orange would lose the value it was
              # carrying. A ring was the earlier answer and 3D has none — Plotly
              # draws no marker outline in a scene — so it left the one view
              # that most needs a landmark without one. The client adds the
              # second trace, since it is the one assembling the coordinates.
              "selection": _selection_meta({0: range(len(trials))},
                                           style=OVERLAY, highlight=1)},
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


def configuration_projection_plot(result, display_metric, method,
                                  config_space=None):
    """Every trial as one point in a projected configuration space.

    The same operation the cube performs, over a larger set of candidates: both
    keep two or three linear coordinates and drop the rest, but the cube is
    restricted to the coordinate axes and this is not. What that buys is a
    subspace chosen to lose as little as possible, on data standardised first —
    so a distance between two points here approximates a real distance between
    two configurations, which is what makes a cluster a cluster, while the
    cube's mixes units and is not a distance at all. See `core.projection`,
    which sets out where the two differ and where they do not.

    Shaped exactly like the cube's payload, and for the same reason: the
    coordinates ride in `customdata`, one column per component, with `x`/`y`
    left empty and the client assembling two or three of them. One payload
    therefore covers both dimensionalities with no round trip, and one piece of
    client code draws all three of this figure's views (see
    experiment_detail.html's applyCubeAxes).

    Returns None when the projection has nothing to say — too few trials, or a
    single hyperparameter — with the reason in `layout.meta["warning"]` of
    nothing at all, since None is the figure's own empty state.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None
    scores = [t.scores[display_metric] for t in trials]
    coordinates, labels, warning = project(config_space, trials, scores, method)
    if warning or not labels:
        return None

    fig = go.Figure(go.Scatter(
        x=[], y=[], mode="markers", customdata=coordinates,
        text=[f"Trial {t.trial}" for t in trials],
        marker=dict(size=8, color=scores, colorscale=_INTENSITY_SCALE, showscale=True,
                    colorbar=dict(title=display_metric.capitalize()),
                    # Opaque, and each point ringed in the card's own colour.
                    # Left to blend, two points that nearly overlap composite
                    # into something darker than either — which on a scale where
                    # darker means better is a trial that did not happen. The
                    # ring makes an overlap read as two points rather than as
                    # one good one. (`opacity` defaults to 1; said out loud
                    # because it is load-bearing here rather than incidental.)
                    opacity=1, line=dict(width=1, color="#FFFFFF")),
    ))
    fig.update_layout(
        meta={"components": labels,
              # No log axes: a principal component is a combination of
              # hyperparameters and has no units to be logarithmic in. The
              # scaling that mattered happened before the projection, in
              # `encode_configurations`.
              "log_hps": [],
              "selection": _selection_meta({0: range(len(trials))},
                                           style=OVERLAY, highlight=1)},
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


#: How many value labels one parallel-coordinates axis carries. Enough to read
#: a position off, few enough that they don't collide on a crowded figure.
_PARCOORDS_TICKS = 5


def _scale_color(scale, position: float) -> str:
    """The colour at *position* (0-1) along a Plotly colourscale.

    Plotly interpolates a colourscale itself when it is handed an array of
    values, but a `go.Scatter` line takes one concrete colour — so with a trace
    per trial (see parallel_coordinates_plot) the interpolation has to happen
    here instead.
    """
    position = min(1.0, max(0.0, position))
    for (lo, low), (hi, high) in zip(scale, scale[1:]):
        if position <= hi:
            t = 0.0 if hi == lo else (position - lo) / (hi - lo)
            return _mix(low, high, t)
    return scale[-1][1]


def _mix(low: str, high: str, t: float) -> str:
    """*low* and *high* as #rrggbb, mixed *t* of the way from one to the other."""
    a = [int(low[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(high[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _axis_positions(values):
    """*values* mapped onto 0-1, and the function that did it.

    Every parallel-coordinates axis has its own units, so they are all drawn on
    one 0-1 axis with their real numbers written beside them as ticks. A
    hyperparameter the search never varied has no range to normalize against and
    sits down the middle.
    """
    lo, hi = min(values), max(values)
    if hi == lo:
        def place(_value):
            return 0.5
    else:
        def place(value):
            return (value - lo) / (hi - lo)
    return [place(v) for v in values], place


def _dimension_ticks(dimension, place):
    """`(position, label)` for one axis's value labels.

    A recoded axis — log or categorical — already carries the tick positions and
    the text to put on them, because the numbers plotted are not the numbers
    meant; those are used as they are. A plain numeric axis gets evenly spaced
    ticks over its own range.
    """
    if "tickvals" in dimension:
        return [(place(v), t)
                for v, t in zip(dimension["tickvals"], dimension["ticktext"])]
    lo, hi = min(dimension["values"]), max(dimension["values"])
    if hi == lo:
        return [(0.5, sigfigs(lo))]
    step = (hi - lo) / (_PARCOORDS_TICKS - 1)
    return [(place(lo + step * i), sigfigs(lo + step * i))
            for i in range(_PARCOORDS_TICKS)]


#: Points drawn along each span between two neighbouring axes. A polyline with
#: one point per axis can only be clicked within a few pixels of an axis, and the
#: whole middle of every span — most of the line — hits nothing.
_PARCOORDS_STEPS = 4


def _densified_length(n_dimensions: int) -> int:
    """How many points one trial's polyline carries. See `_densified`."""
    return _PARCOORDS_STEPS * (n_dimensions - 1) + 1 if n_dimensions > 1 else 1


def _densified(values):
    """One trial's polyline as `(x, y)`, with points along each span.

    The line itself is unchanged — the extra points sit on the straight segment
    between two axes — but they are what a click can land on. Every polyline is
    the same length, which is what lets the selection contract turn a flat point
    index back into a trial by dividing (see `_selection_meta`'s `segment`).
    """
    if len(values) < 2:
        return [0], list(values)
    x, y = [], []
    for i in range(len(values) - 1):
        for step in range(_PARCOORDS_STEPS):
            t = step / _PARCOORDS_STEPS
            x.append(i + t)
            y.append(values[i] + (values[i + 1] - values[i]) * t)
    x.append(len(values) - 1)
    y.append(values[-1])
    return x, y


def parallel_coordinates_plot(result, display_metric, config_space=None):
    """Every trial as one line across its hyperparameters, ending at its
    *display_metric* score — DeepCave's best tool for spotting hyperparameter
    interactions at a glance: a cluster of high-scoring lines that all bend
    through the same region of one axis says that axis matters.

    Axes are ordered by HyperSHAP tunability (most important first, the same
    "most important first" convention the importance bar/interactions heatmap
    already use) rather than DeepCave's own fANOVA ordering — this app's
    explanations are HyperSHAP's throughout, so the ordering should be too.
    Falls back to config order (unordered) when importance isn't available
    for this metric (e.g. HyperSHAP failed, or a result predating Phase 1a).

    **Drawn from scatter traces rather than `go.Parcoords`.** A trial is
    selectable here, like it is on every other figure that draws all of them,
    and the selected trial's line is thicker and drawn over the rest. Parcoords
    emits no click, has no per-line width and no way to reorder what sits on
    top, so none of that is reachable through it. What it costs is Parcoords'
    own two gifts: dragging an axis to reorder it, and dragging along one to
    filter the lines through it. One trace per trial, since a Scatter line takes
    a single colour and colouring by score is the point of the figure.

    A non-numeric hyperparameter (categorical, or boolean — bool is a numeric
    subtype in Python, but "True"/"False" reads better as a category than as
    0/1 here) is recoded to its sorted-unique values' integer position, with
    the original values written on the ticks.

    A hyperparameter searched logarithmically is recoded too, for a different
    reason: every axis here shares one 0-1 scale, so `C` on the SVM model —
    sampled log-uniformly over 0.01 to 100 — would pack three quarters of its
    lines into the bottom tenth of its axis, which is the opposite of what this
    figure is for. Its values go on as `log10` and the ticks are relabelled with
    the native numbers. This is what DeepCAVE does throughout
    (`run.get_encoded_data` plus `get_hyperparameter_ticks`); here it is
    confined to the axes that need it, so every linear axis keeps showing its
    own values directly.

    Returns None with no trials.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None
    hp_names = list(trials[0].config.keys())
    importance = result.hyperparameter_importance.get(display_metric, {})
    ordered = sorted(hp_names, key=lambda h: importance.get(h, 0.0), reverse=True)

    log_hps = log_hyperparameters(config_space, ordered)

    dimensions = []
    for hp in ordered:
        raw = [t.config.get(hp) for t in trials]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
            if hp in log_hps and all(v > 0 for v in raw):
                lo, hi = math.log10(min(raw)), math.log10(max(raw))
                ticks = [lo + (hi - lo) * i / 4 for i in range(5)] if hi > lo else [lo]
                dimensions.append(dict(
                    label=hp, values=[math.log10(v) for v in raw],
                    tickvals=ticks, ticktext=[sigfigs(10 ** t) for t in ticks],
                ))
            else:
                dimensions.append(dict(label=hp, values=raw))
        else:
            uniques = sorted(set(raw), key=str)
            index = {v: i for i, v in enumerate(uniques)}
            dimensions.append(dict(
                label=hp, values=[index[v] for v in raw],
                tickvals=list(range(len(uniques))), ticktext=[str(v) for v in uniques],
            ))

    scores = [t.scores[display_metric] for t in trials]
    dimensions.append(dict(label=display_metric.capitalize(), values=scores))

    columns, ticks, shapes = [], [], []
    for x, dimension in enumerate(dimensions):
        placed, place = _axis_positions(dimension["values"])
        columns.append(placed)
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1,
                           line=dict(color="rgba(140,140,140,0.55)", width=1)))
        # Offset in pixels, not in axis units: the tick sits beside its axis at
        # any width, and the annotation's own x stays the axis it belongs to.
        ticks.extend(dict(x=x, y=y, text=text, showarrow=False, xshift=-5,
                          xanchor="right", font=dict(size=9, color="#6b6b6b"))
                     for y, text in _dimension_ticks(dimension, place))

    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0

    fig = go.Figure()
    # Worst first, so the best trials are added last and draw over the rest.
    # Trace order is z-order and nothing can change it afterwards, so the
    # ordering is a decision made here: with hundreds of lines crossing, the
    # ones worth following are the ones that should be on top, and in trial
    # order they were simply whichever ran latest.
    by_score = sorted(range(len(trials)), key=lambda i: scores[i])
    for i in by_score:
        trial = trials[i]
        x, y = _densified([column[i] for column in columns])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=_scale_color(_INTENSITY_SCALE, (scores[i] - lo) / span),
                      width=1.5),
            # Invisible but present: a lines-only trace takes no clicks at all,
            # and this is a figure you click.
            marker=dict(size=5, opacity=0.01),
            name=f"Trial {trial.trial}", showlegend=False,
            hovertemplate=f"Trial {trial.trial}<br>{display_metric} "
                          f"{sigfigs(scores[i])}<extra></extra>",
        ))
    # Nothing drawn, only a colour bar: with a colour per trace there is no
    # array for Plotly to build one from, so one trace carries the scale.
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="markers", showlegend=False,
        hoverinfo="skip",
        marker=dict(color=[lo, hi], colorscale=_INTENSITY_SCALE, cmin=lo, cmax=hi,
                    showscale=True, opacity=0,
                    colorbar=dict(title=display_metric.capitalize())),
    ))
    # Last, so it draws over every trial line. Empty until something is
    # selected; the page fills it from whichever trial's trace (see
    # experiment_detail.html's applySelection).
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=SELECTION_COLOR, width=4),
    ))
    fig.update_layout(
        margin=dict(t=40, b=20, l=40, r=20), shapes=shapes, annotations=ticks,
        xaxis=dict(tickmode="array", tickvals=list(range(len(dimensions))),
                   ticktext=[d["label"] for d in dimensions],
                   range=[-0.35, len(dimensions) - 0.65],
                   showgrid=False, zeroline=False, side="top"),
        yaxis=dict(visible=False, range=[-0.06, 1.06], fixedrange=True),
        # Generous, because what is being aimed at is a line and the points that
        # carry the click are invisible.
        hoverdistance=25,
        # Trace order is score order, not trial order, so which trial a trace
        # stands for has to be said rather than assumed.
        meta={"selection": _selection_meta(
            {trace: [trial] for trace, trial in enumerate(by_score)},
            style=LINE, segment=_densified_length(len(dimensions)),
            # Which trace the highlight is drawn into — the one added last.
            highlight=len(trials) + 1)},
    )
    return fig


def partial_dependence_plot(hp_name, grid, ice_lines, pdp):
    """DeepCave's PDP/ICE plot for one hyperparameter: every trial's
    Individual Conditional Expectation curve (thin, translucent, batched
    into one trace with `None`-separated segments so trial count doesn't
    multiply the trace count) plus the bold Partial Dependence mean curve on
    top — ICE lines that all move together say this hyperparameter doesn't
    interact with anything else; ICE lines that fan out somewhere on the
    grid say it does, right where they fan out.

    x is *grid* on a categorical axis (`xaxis_type="category"`) even for a
    numeric hyperparameter's evenly-spaced grid — visually identical for an
    even grid, and it means a categorical hyperparameter's grid (strings)
    never has to be handled as a special case here.

    Returns None when *grid* is empty (too few trials to fit a surrogate, or
    no valid configuration anywhere on the grid — see
    BaseOptimizer.compute_partial_dependence).
    """
    if not grid:
        return None
    ice_x, ice_y = [], []
    for row in ice_lines:
        ice_x.extend(grid)
        ice_x.append(None)
        ice_y.extend(row)
        ice_y.append(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ice_x, y=ice_y, mode="lines", line=dict(width=1, color=MARKER_COLOR),
        opacity=0.25, name="Individual trials (ICE)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=pdp, mode="lines+markers", name="Partial dependence",
        line=dict(width=3, color=ACCENT_COLOR),
    ))
    fig.update_layout(
        xaxis_title=hp_name, xaxis_type="category", yaxis_title="Predicted score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def local_effects_plot(hp_names: list, rows: list):
    """Beeswarm of every sampled trial's local ablation: one row per
    hyperparameter, one point per trial, placed by what that hyperparameter's
    value did for that trial against the config-space default.

    The one view that shows *spread*. Every other explanation on this page is a
    single number per hyperparameter — an average over the run, or one trial's.
    Neither can distinguish a hyperparameter that reliably helps by a little
    from one that helps enormously in half the space and hurts in the other,
    and those call for opposite decisions.

    Rows are ordered by spread, widest first, because a wide row is the
    interesting finding here — the opposite of the "biggest average first"
    ordering the other views use, and deliberately so.

    Points are jittered vertically, deterministically by position within the
    row, so overlapping values stay countable. Colour follows sign, matching
    the waterfall.

    Returns None when there is nothing to plot.
    """
    if not hp_names or not rows:
        return None

    spread = {}
    for name in hp_names:
        values = [row["effects"].get(name, 0.0) for row in rows]
        spread[name] = max(values) - min(values) if values else 0.0
    ordered = sorted(hp_names, key=lambda n: spread[n])

    fig = go.Figure()
    for index, name in enumerate(ordered):
        values = [row["effects"].get(name, 0.0) for row in rows]
        # Deterministic jitter: a random one would make the figure move between
        # reloads for no information, and these are already sampled evenly
        # through the run so position carries nothing.
        offsets = [((i % 7) - 3) / 18.0 for i in range(len(values))]
        fig.add_trace(go.Scatter(
            x=values, y=[index + o for o in offsets], mode="markers",
            marker=dict(size=7, opacity=0.65,
                        color=[MARKER_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]),
            text=[f"Trial {row['trial']}<br>{name} {_signed(v)}"
                  for row, v in zip(rows, values)],
            hoverinfo="text", showlegend=False,
        ))
    fig.add_vline(x=0, line=dict(color="rgba(140,140,140,0.6)", width=1))
    # A rule between rows. The points are jittered vertically so that
    # overlapping values stay countable, which without a divider leaves a
    # scatter whose rows have to be told apart by the tick labels alone —
    # exactly where a wide row (the interesting finding here) is widest.
    for index in range(1, len(ordered)):
        fig.add_hline(y=index - 0.5,
                      line=dict(color="rgba(140,140,140,0.25)", width=1))
    fig.update_layout(
        xaxis_title="Effect vs. default  (left: hurt, right: helped)",
        yaxis=dict(tickmode="array", tickvals=list(range(len(ordered))), ticktext=ordered),
        margin=dict(t=20, b=40, l=110, r=20), showlegend=False,
        # One trace per row, and the selected trial has a point in every one of
        # them — reading a hyperparameter's effect for a trial means finding
        # that trial on each row, which is exactly what the highlight saves you
        # doing by eye. The rows are a sample, so which trial each point belongs
        # to has to be carried rather than inferred from its position.
        meta={"selection": _selection_meta(
            {i: [row["index"] for row in rows] for i in range(len(ordered))},
            style=OUTLINE)},
    )
    return fig


def trial_duration_plot(result):
    """Bar of each trial's evaluation duration (seconds). Metric-independent.

    Returns None when there are no trials.
    """
    trials = result.trials
    if not trials:
        return None
    fig = go.Figure(go.Bar(
        x=[t.trial for t in trials], y=[t.duration for t in trials],
        marker_color=MARKER_COLOR,
    ))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Duration (s)",
        margin=dict(t=20, b=40, l=40, r=20), showlegend=False,
        meta={"selection": _selection_meta({0: range(len(trials))},
                                           style=RECOLOR)},
    )
    return fig
