"""`plots` holds one Plotly builder per drawn figure, named for its `catalog.py`
figure, such as `performance_over_time_plot`. The builders are pure functions of
an OptimizationResult and a metric that return a Figure or None, with no request
state. A per-metric builder returns None when any trial lacks a score for the
metric (see `OptimizationResult.has_every_score`).
"""

import math
from typing import Any, cast

import plotly.graph_objects as go

from core.metrics import to_error
from core.projection import fit, log_hyperparameters

from ..formatting import sigfigs

# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _

# Each palette colour means one thing on every figure, by convention only.
# `MARKER_COLOR` marks trials and positive signed values, `ACCENT_COLOR` marks
# the best so far, `NEGATIVE_COLOR` marks negative signed values, and
# `SELECTION_COLOR` marks the selected trial alone. `MARKER_COLOR` and
# `SELECTION_COLOR` are public because the page restyles the selection itself
# (see `selection_colors` in ui/views/dashboard.py).
MARKER_COLOR = "#636EFA"
#: `SELECTION_COLOR` is the soft coral that `#EF553B` became when faded over
#: Plotly's background, painted at full opacity on every figure so the selection
#: looks the same everywhere. `--selected` in app.css holds the same value.
SELECTION_COLOR = "#EC8273"
ACCENT_COLOR = "#10B981"
NEGATIVE_COLOR = "#FF2B2B"
#: A proposed configuration from the cube, which nobody has run, gets its own
#: colour so it cannot pass for a trial. `--proposal` in app.css has the same
#: value.
PROPOSAL_COLOR = "#8B5CF6"

#: The uncertainty field uses a violet ramp, a hue the palette leaves unused,
#: since green means the best so far and blue already encodes the metric on the
#: same figure. Both ends are translucent so the field stays scenery, and the
#: ramp rises with uncertainty.
UNCERTAINTY_SCALE = [
    [0.0, "rgba(124, 58, 237, 0.02)"],
    [1.0, "rgba(124, 58, 237, 0.42)"],
]


# A failed trial is drawn as an `x-thin` cross in `NEGATIVE_COLOR`, and
# `marker.line` gives the unfilled glyph its visibility and weight. A selected
# failure keeps its cross and takes the selection colour.
FAILURE_SYMBOL = "x-thin"
FAILURE_LINE_WIDTH = 3
#: Thinner on the cube and the projection, whose ringed markers would make a
#: full-weight cross the loudest mark.
FAILURE_LINE_WIDTH_RINGED = 2
#: Large enough to read as a cross.
FAILURE_SIZE = 11
#: Tint for failures drawn as fills, such as duration bars.
FAILURE_FILL = "rgba(255, 43, 43, 0.45)"


def _translucent(color: str, alpha: float = 0.7) -> str:
    """`_translucent` lets a trace fade individual points and keep a failure's
    cross at full strength.
    """
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _failure_marks(trials, *, size, symbol="circle", line_width=1,
                   line_color="#FFFFFF", selected_idx=None,
                   failure_width=FAILURE_LINE_WIDTH):
    """`_failure_marks` returns marker keys that mark failed trials, or `{}`
    when none failed. It styles the existing trace instead of adding one, since
    a failed trial must stay selectable and `_selection_meta` maps trace
    positions to trials one for one.
    """
    failed = [t.failed for t in trials]
    if not any(failed):
        return {}

    sizes = list(size) if isinstance(size, (list, tuple)) else [size] * len(trials)
    widths = list(line_width) if isinstance(line_width, (list, tuple)) else [line_width] * len(trials)
    colors = list(line_color) if isinstance(line_color, (list, tuple)) else [line_color] * len(trials)

    for i, bad in enumerate(failed):
        if not bad:
            continue
        sizes[i] = max(sizes[i], FAILURE_SIZE)
        widths[i] = failure_width
        colors[i] = (SELECTION_COLOR if i == selected_idx else NEGATIVE_COLOR)

    return {
        "symbol": [FAILURE_SYMBOL if bad else symbol for bad in failed],
        "size": sizes,
        "line": {"width": widths, "color": colors},
    }


#: Selection styles; the figure's colour meaning decides which applies.
RECOLOR = "recolor"   # free colour: paint the selected point
OUTLINE = "outline"   # colour carries meaning: ring it
LINE = "line"         # polyline per trial (parallel coordinates)
OVERLAY = "overlay"   # selected point redrawn on top


def _signed(value) -> str:
    """`_signed` always writes the sign, since helped or hurt is the reading
    here.
    """
    return f"{'+' if value >= 0 else '-'}{sigfigs(abs(value))}"


def _selection_meta(by_trace: dict, *, style, segment=1, **extra) -> dict:
    """`_selection_meta` tells the page's selection script which trial each
    clicked point is, via `layout.meta["selection"]`. *by_trace* maps a trace
    index to positions in `result.trials`, because traces outside the map (an
    incumbent line, a colour-bar carrier) carry no trials and local effects
    draws a sample. *segment* counts points per trial, 1 except in parallel
    coordinates, so the trial at point *p* of trace *t* is `by_trace[t][p //
    segment]`.
    """
    return dict(trials={str(k): list(v) for k, v in by_trace.items()},
                style=style, segment=segment, **extra)


def incumbent_scores(result, display_metric):
    """`incumbent_scores` is a thin wrapper over the memoized
    `OptimizationResult.incumbent_scores`, kept for the `ui.figures` exports.
    """
    return result.incumbent_scores(display_metric)


def performance_over_time_plot(result, display_metric, *, x_axis="trial",
                                y_axis="score", selected_idx=None):
    """*x_axis* ("trial" | "time") and *y_axis* ("score" | "error") pick one of
    four views of one curve. Time is cumulative trial duration, so gaps between
    resumed runs never appear. The error view is floored at 1e-3 on a log axis,
    and the incumbent is drawn as a step line with diamonds at each improvement.
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

    metric = result.metric(display_metric)
    incumbents = incumbent_scores(result, display_metric)
    if y_axis == "error":
        errors = [to_error(metric, t.scores[display_metric]) for t in trials]
        # An unbounded higher-is-better metric has no error to plot (see
        # `core.metrics.to_error`), so the figure declines.
        if any(e is None for e in errors):
            return None
        incumbent_errors = [to_error(metric, v) for v in incumbents]
        # The 1e-3 floor applies only on a log axis, since an error metric can
        # be negative.
        y_type = "log" if all(e > 0 for e in cast(list[float], errors)) else "linear"
        floor = (lambda e: max(1e-3, e)) if y_type == "log" else (lambda e: e)
        ys = [floor(e) for e in errors]
        incumbent_ys = [floor(e) for e in incumbent_errors]
        y_title, outcome_name = "Error", "Trial error"
    else:
        ys = [t.scores[display_metric] for t in trials]
        incumbent_ys = incumbents
        y_title, y_type, outcome_name = metric.title, "linear", "Trial score"

    colors = [MARKER_COLOR] * len(trials)
    sizes = [6] * len(trials)
    if selected_idx is not None and 0 <= selected_idx < len(trials):
        colors[selected_idx] = SELECTION_COLOR
        sizes[selected_idx] = 13

    # In the metric's direction: lower RMSE improves.
    improved = [i == 0 or metric.better(incumbents[i], incumbents[i - 1])
                for i in range(len(trials))]

    # Opacity is set per point so a failure's cross and the selected point stay
    # at full strength while the rest fade.
    fills = [c if i == selected_idx else _translucent(c)
             for i, c in enumerate(colors)]
    marker = dict(size=sizes, color=fills)
    marker.update(_failure_marks(trials, size=sizes, selected_idx=selected_idx))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=outcome_name, marker=marker,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=incumbent_ys, mode="lines", name=_("Incumbent"),
        line=dict(width=2, shape="hv", color=ACCENT_COLOR),
        # The incumbent line skips hover, since under closest hover it answered
        # for nearby trials and a click named nothing.
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[xs[i] for i in range(len(trials)) if improved[i]],
        y=[incumbent_ys[i] for i in range(len(trials)) if improved[i]],
        mode="markers", name=_("New incumbent"),
        marker=dict(size=9, color=ACCENT_COLOR, symbol="diamond"),
        # The diamonds sit over the trials they mark, so skipping hover lets the
        # trial beneath answer clicks.
        hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=y_title, yaxis_type=y_type,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
        # Trace 0 only, since incumbent traces hold no trials.
        meta={"selection": _selection_meta({0: range(len(trials))},
                                           style=RECOLOR)},
    )
    return fig


def hyperparameter_importance_plot(importance: dict, rendering: str = "pie",
                                   title: str = "Importance"):
    """`hyperparameter_importance_plot` draws any HyperSHAP game's `{hp:
    weight}` dict as a pie or vertical bar. It returns None for an empty dict or
    the "table" rendering, which the template draws.
    """
    if not importance or rendering == "table":
        return None
    if rendering == "bar":
        # Bars sort by weight, largest on the left.
        names, values = zip(*sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=MARKER_COLOR))
        fig.update_layout(yaxis_title=title,
                          margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        return fig
    fig = go.Figure(go.Pie(
        labels=list(importance.keys()), values=list(importance.values()),
        textinfo="label+percent", textposition="inside",
        insidetextorientation="horizontal", textfont=_TEXT_FONT,
        # `hole=0` is explicit, since `Plotly.react` treats an absent attribute
        # more weakly than a default.
        hole=0,
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
                      **_UNIFORM_TEXT)
    return fig


#: `textfont` fixes one label size for every wedge, and `uniformtext` hides
#: labels that no longer fit, leaving them in the hover. `minsize` equals the
#: font size, which keeps Plotly from scaling labels again.
_TEXT_SIZE = 12
_TEXT_FONT = {"size": _TEXT_SIZE}
_UNIFORM_TEXT: dict[str, Any] = {"uniformtext": {"minsize": _TEXT_SIZE, "mode": "hide"}}

#: One hue at two intensities, since banked and remaining are parts of one
#: quantity. The darker shade is the gain still to take.
_PROGRESS_SHADES = (MARKER_COLOR, "#9DA5F5")


def hyperparameter_progress_plot(rows: list, rendering: str = "pie"):
    """`hyperparameter_progress_plot` splits each hyperparameter's achievable
    gain into banked and still to gain, from `core.analytics.tuning_progress`
    rows. The bar rendering stacks banked under the remainder, and the pie gives
    each hyperparameter two adjacent slices. Banked is clamped into
    0..achievable for drawing, since the two estimates can disagree, and the
    table keeps the signed value. The function returns None with no rows or for
    the "table" rendering.
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
        # Banked first, so the bar fills up from the base.
        for values, label, shade in ((banked, _("Already banked"), _PROGRESS_SHADES[1]),
                                     (left, _("Still to gain"), _PROGRESS_SHADES[0])):
            fig.add_trace(go.Bar(x=names, y=values, name=label, marker_color=shade,
                                 hovertemplate="%{x}<br>" + label
                                               + " %{y:.4g}<extra></extra>"))
        fig.update_layout(
            barmode="stack", yaxis_title=_("Achievable gain"),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1))
        return fig

    # A flat pie gives each hyperparameter two adjacent slices and avoids a
    # sunburst's empty centre.
    labels, values, colours, hovers, captions = [], [], [], [], []
    for name, taken, remaining in zip(names, banked, left):
        for part, value, shade in (("banked", taken, _PROGRESS_SHADES[1]),
                                   ("still to gain", remaining, _PROGRESS_SHADES[0])):
            labels.append(f"{name} — {part}")
            values.append(value)
            colours.append(shade)
            hovers.append(f"{name}<br>{part} {sigfigs(value)}")
        # Only the roomier part of each pair carries the name.
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
    """`hyperparameter_ablation_plot` draws one trial's HyperSHAP ablation as a
    waterfall from the default, ordered by magnitude, so the running total shows
    how far the trial got. Steps are signed, and the axis is relative to the
    default, whose own score the game omits. The function returns None for an
    empty *ablation* (see core.analytics.compute_hp_ablation).
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
    fig.update_layout(yaxis_title=_("vs. default"),
                      margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
    return fig


def hyperparameter_interactions_heatmap_plot(interactions: dict):
    """The diagonal holds each hyperparameter's signed order-1 value, and
    off-diagonal cells hold pairwise interactions, positive for synergy and
    negative for redundancy. Rows are ordered by absolute diagonal value,
    largest first. `hyperparameter_interactions_heatmap_plot` returns None for
    empty *interactions*.
    """
    if not interactions:
        return None
    params = sorted(interactions.keys(), key=lambda p: abs(interactions[p][p]), reverse=True)
    z = [[interactions[a][b] for b in params] for a in params]
    fig = go.Figure(go.Heatmap(
        z=z, x=params, y=params, colorscale=_SIGN_SCALE, zmid=0,
        colorbar=dict(title=_("Interaction")),
    ))
    fig.update_layout(margin=dict(t=20, b=40, l=80, r=20))
    return fig


def hyperparameter_interactions_bar_plot(interactions: dict, top_k: int = 10):
    """`hyperparameter_interactions_bar_plot` ranks the *top_k* strongest pairs
    by magnitude, keeping the sign as colour. The function returns None with
    fewer than two hyperparameters.
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
    fig.update_layout(yaxis_title=_("Interaction strength"),
                      margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    return fig


#: Score figures use one blue ramp by intensity, since a score has a good end
#: and a bad end with no meaningful middle.
_INTENSITY_SCALE = [
    [0.00, "#EEF1FE"],
    [0.25, "#C3CBFA"],
    [0.50, "#93A0F4"],
    [0.75, "#6373EC"],
    [1.00, "#2B3AA8"],
]

#: Synergy and redundancy reuse the sign colours.
_SYNERGY_COLOR = MARKER_COLOR
_REDUNDANCY_COLOR = NEGATIVE_COLOR

#: `_SIGN_SCALE` is built here because Plotly's "RdBu" resolves differently in
#: plotly.min.js, where it put positive values on red.
_SIGN_SCALE = [
    [0.0, NEGATIVE_COLOR],
    [0.5, "#F7F7F7"],
    [1.0, MARKER_COLOR],
]


def _circle_positions(names: list) -> dict:
    """`nx.circular_layout` places HyperSHAP's SI graph on a circle, and cos and
    sin reproduce that layout natively in Plotly.
    """
    import math

    n = len(names)
    return {name: (math.sin(2 * math.pi * i / n), math.cos(2 * math.pi * i / n))
            for i, name in enumerate(names)}


def hyperparameter_graph_plot(moebius: list, top_k: int = 15):
    """`hyperparameter_graph_plot` draws size-2 Möbius terms as edges and larger
    terms as hubs with spokes, as shapiq does, with order-1 terms sizing the
    nodes. Width shows magnitude and colour shows sign. The graph is the only
    view above order 2, which is why the Möbius decomposition is stored. Only
    *top_k* terms are drawn, since a 10-hyperparameter run has 1,013.
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

    # Sorted here instead of trusting the caller's ranking.
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
    # The hubs carry hover text, since a `shape` has none.
    fig.add_trace(go.Scatter(
        x=hub_x, y=hub_y, mode="markers", name=_("Interaction"),
        marker=dict(size=9, color="rgba(0,0,0,0)",
                    line=dict(width=1, color="rgba(120,120,120,0.5)")),
        text=hub_text, hoverinfo="text", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names],
        mode="markers+text", name=_("Hyperparameter"),
        marker=dict(size=[14 + 26 * own.get(n, 0.0) / biggest_own for n in names],
                    color=MARKER_COLOR, opacity=0.85),
        text=names, textposition="top center",
        hovertext=[f"{n}<br>on its own {sigfigs(own.get(n, 0.0))}" for n in names],
        hoverinfo="text", showlegend=False,
    ))
    # The axes skip `scaleanchor`, since a 1:1 ratio would shrink the circle on
    # a wide card and node distances mean nothing. The x range is wider to leave
    # room for side labels.
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.25, 1.3]),
        margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
    )
    return fig


def hyperparameter_upset_plot(moebius: list, top_k: int = 12):
    """`hyperparameter_upset_plot` ranks the strongest Möbius coalitions as
    signed bars over a membership matrix, and stays legible past the six
    hyperparameters where the graph stops. The function returns None without a
    coalition of two or more.
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

    # Absent dots draw first, and one rule per column links its members.
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

    fig.update_yaxes(title_text=_("Interaction"), row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=names, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(margin=dict(t=20, b=20, l=90, r=20), showlegend=False)
    return fig


#: Every order from 3 up shares one band, since past pairs the distinction stops
#: being actionable.
_ORDER_BANDS = (
    (1, 1, "On its own"),
    (2, 2, "In pairs"),
    (3, None, "In larger groups"),
)


def hyperparameter_orders_plot(moebius: list):
    """`hyperparameter_orders_plot` stacks each hyperparameter's influence by
    source: itself, pairs, or larger groups. It reads the Möbius decomposition
    and splits each term's magnitude equally among its members, since signed
    shares would cancel. Bars are ordered by total height.
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
        barmode="stack", yaxis_title=_("Attributed influence"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def _trial_hovers(result, display_metric):
    """The hover lists number, score and full configuration, since projected
    coordinates name no values."""
    title = result.metric(display_metric).title
    return [f"Trial {t.trial}<br>{title} = {sigfigs(t.scores.get(display_metric))}<br>"
            + "<br>".join(f"{k} = {sigfigs(v)}" for k, v in sorted(t.config.items()))
            for t in result.trials]


def configuration_cube_plot(result, display_metric, config_space=None):
    """`configuration_cube_plot` draws each trial at its actual hyperparameter
    values, coloured by score. The axis pickers start empty, and the trace
    therefore ships every hyperparameter in `customdata` (columns in
    `layout.meta["hp_names"]`) for the browser to assign axes (see
    applyCubeAxes). Log-searched hyperparameters are listed in
    `layout.meta["log_hps"]` for log axes. The function returns None with no
    trials or hyperparameters, and one hyperparameter is enough.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None
    hp_names = list(trials[0].config.keys())
    if not hp_names:
        return None
    customdata = [[t.config.get(h) for h in hp_names] for t in trials]
    scores = [t.scores[display_metric] for t in trials]
    cube_marker = dict(size=8, color=scores, colorscale=_INTENSITY_SCALE,
                       # Darker is better; only the ramp direction flips.
                       reversescale=not result.metric(display_metric).higher_is_better,
                       showscale=True,
                       colorbar=dict(title=result.metric(display_metric).title),
                       # Points are opaque and ringed in the card's colour,
                       # since a blended overlap would look darker and read as a
                       # better trial.
                       opacity=1, line=dict(width=1, color="#FFFFFF"))
    # A failure's fill sits at 0.0, the worst end, under its cross.
    cube_marker.update(_failure_marks(trials, size=8,
                                      failure_width=FAILURE_LINE_WIDTH_RINGED))
    fig = go.Figure(go.Scatter(
        x=[], y=[], mode="markers", customdata=customdata,
        text=[f"Trial {t.trial}" for t in trials], marker=cube_marker,
        hovertext=_trial_hovers(result, display_metric), hoverinfo="text",
    ))
    fig.update_layout(
        meta={"hp_names": hp_names,
              "log_hps": log_hyperparameters(config_space, hp_names),
              # The client redraws the selected trial on top in the selection
              # colour, since recolouring would hide the score and 3D scenes
              # draw no rings.
              "selection": _selection_meta({0: range(len(trials))},
                                           style=OVERLAY, highlight=1)},
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


def configuration_projection_plot(result, display_metric, method,
                                  config_space=None):
    """`configuration_projection_plot` projects trials onto a standardised
    subspace chosen to lose least, so distances approximate real ones (see
    `core.projection`). Coordinates ride in `customdata` like the cube's, and
    the client builds two or three of them. The function returns None when the
    projection fails, with too few trials or a single hyperparameter.
    """
    trials = result.trials
    if not trials or not result.has_every_score(display_metric):
        return None
    scores = [t.scores[display_metric] for t in trials]
    fitted, _warning = fit(config_space, trials, scores, method)
    if fitted is None or not fitted.labels:
        return None
    coordinates, labels = fitted.coordinates, fitted.labels

    marker = dict(size=8, color=scores, colorscale=_INTENSITY_SCALE,
                  # Same ramp as the cube.
                  reversescale=not result.metric(display_metric).higher_is_better,
                  showscale=True,
                  colorbar=dict(title=result.metric(display_metric).title),
                  # Opaque and ringed for the same reason as the cube.
                  opacity=1, line=dict(width=1, color="#FFFFFF"))
    marker.update(_failure_marks(trials, size=8,
                                 failure_width=FAILURE_LINE_WIDTH_RINGED))
    fig = go.Figure(go.Scatter(
        x=[], y=[], mode="markers", customdata=coordinates,
        text=[f"Trial {t.trial}" for t in trials], marker=marker,
        hovertext=_trial_hovers(result, display_metric), hoverinfo="text",
    ))
    fig.update_layout(
        meta={"components": labels,
              # No log axes, since components have no units.
              "log_hps": [],
              # The linear map lets the page place a proposed configuration (see
              # `core.projection.Fitted.linear_map`).
              "map": fitted.linear_map(),
              "selection": _selection_meta({0: range(len(trials))},
                                           style=OVERLAY, highlight=1)},
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


#: Value labels per axis.
_PARCOORDS_TICKS = 5


def _scale_color(scale, position: float) -> str:
    """A `go.Scatter` line takes one concrete colour, so per-trial traces
    interpolate the scale here.
    """
    position = min(1.0, max(0.0, position))
    for (lo, low), (hi, high) in zip(scale, scale[1:]):
        if position <= hi:
            t = 0.0 if hi == lo else (position - lo) / (hi - lo)
            return _mix(low, high, t)
    return scale[-1][1]


def _mix(low: str, high: str, t: float) -> str:
    a = [int(low[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(high[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _axis_positions(values):
    """Each axis maps onto 0-1 with its real values as ticks, and an unvaried
    hyperparameter sits at 0.5.
    """
    lo, hi = min(values), max(values)
    def place(value):
        return 0.5 if hi == lo else (value - lo) / (hi - lo)

    return [place(v) for v in values], place


def _dimension_ticks(dimension, place):
    """`_dimension_ticks` uses the ticks that log and categorical axes bring,
    and gives plain numeric axes evenly spaced ones.
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


#: `_PARCOORDS_STEPS` adds points along each span, so a click anywhere on a line
#: lands.
_PARCOORDS_STEPS = 4


def _densified_length(n_dimensions: int) -> int:
    return _PARCOORDS_STEPS * (n_dimensions - 1) + 1 if n_dimensions > 1 else 1


def _densified(values):
    """The extra points sit on the straight segments, and equal lengths let
    `_selection_meta`'s `segment` map a point index to a trial.
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
    """`parallel_coordinates_plot` draws each trial as a line across its
    hyperparameters ending at its score. Axes follow HyperSHAP tunability, else
    config order. Scatter traces replace `go.Parcoords`, which offers no click,
    line width or z-order, and categorical and log-searched axes are recoded
    with native tick labels. Failed trials are left out, and the function
    returns None when no trial succeeded.
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
    dimensions.append(dict(label=result.metric(display_metric).title, values=scores))

    columns, ticks, shapes = [], [], []
    for x, dimension in enumerate(dimensions):
        placed, place = _axis_positions(dimension["values"])
        columns.append(placed)
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1,
                           line=dict(color="rgba(140,140,140,0.55)", width=1)))
        # A pixel offset keeps the tick beside its axis at any width.
        ticks.extend(dict(x=x, y=y, text=text, showarrow=False, xshift=-5,
                          xanchor="right", font=dict(size=9, color="#6b6b6b"))
                     for y, text in _dimension_ticks(dimension, place))

    # Failures are left out, since a placeholder score would run a line across
    # every other one and stretch the colour scale. Other figures mark failures
    # without redrawing.
    live = [i for i in range(len(trials)) if not trials[i].failed]
    if not live:
        return None
    lo, hi = min(scores[i] for i in live), max(scores[i] for i in live)
    span = (hi - lo) or 1.0
    metric = result.metric(display_metric)

    def shade(score):
        """`shade` gives the ramp position, 0 worst to 1 best, for both colour
        and z-order.
        """
        along = (score - lo) / span
        return along if metric.higher_is_better else 1.0 - along

    fig = go.Figure()
    # The worst trials are added first, so the best lines draw on top, since
    # trace order is z-order.
    by_score = sorted(live, key=lambda i: shade(scores[i]))
    for i in by_score:
        trial = trials[i]
        x, y = _densified([column[i] for column in columns])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=_scale_color(_INTENSITY_SCALE, shade(scores[i])),
                      width=1.5),
            # Near-invisible markers make the line clickable.
            marker=dict(size=5, opacity=0.01),
            name=_("Trial %(n)s") % {"n": trial.trial}, showlegend=False,
            hovertemplate=(_("Trial %(n)s") % {"n": trial.trial})
                          + f"<br>{display_metric} "
                          + f"{sigfigs(scores[i])}<extra></extra>",
        ))
    # A markerless trace carries the colour bar.
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="markers", showlegend=False,
        hoverinfo="skip",
        marker=dict(color=[lo, hi], colorscale=_INTENSITY_SCALE, cmin=lo, cmax=hi,
                    reversescale=not metric.higher_is_better,
                    showscale=True, opacity=0,
                    colorbar=dict(title=result.metric(display_metric).title)),
    ))
    # The highlight trace comes last and the page fills it.
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
        # Generous hover distance for invisible click points.
        hoverdistance=25,
        # Trace order is score order, so each trace names its trial.
        meta={"selection": _selection_meta(
            {trace: [trial] for trace, trial in enumerate(by_score)},
            style=LINE, segment=_densified_length(len(dimensions)),
            # The highlight index counts drawn traces, since failures add none.
            highlight=len(by_score) + 1)},
    )
    return fig


def partial_dependence_plot(hp_name, grid, ice_lines, pdp):
    """`partial_dependence_plot` draws every trial's ICE curve in one
    `None`-separated trace, with the bold PD mean on top. The x axis is
    categorical, which handles categorical grids with no special case. The
    function returns None for an empty *grid* (see
    core.analytics.compute_partial_dependence).
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
        opacity=0.25, name=_("Individual trials (ICE)"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=pdp, mode="lines+markers", name=_("Partial dependence"),
        line=dict(width=3, color=ACCENT_COLOR),
    ))
    fig.update_layout(
        xaxis_title=hp_name, xaxis_type="category", yaxis_title=_("Predicted score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


#: Amber marks what a person asserted, a hue no measured quantity uses, wherever
#: priors appear.
PRIOR_COLOR = "#B26315"

#: Band σ multiples, widest first so narrower bands draw over.
SIGMA_STEPS = (3, 2, 1)
_BAND_ALPHA = {3: 0.14, 2: 0.22, 1: 0.32}

#: The implied curve gets one faint band, since it infers a surrogate nobody
#: fitted.
FICTION_SIGMA = 1
_GHOST_ALPHA = 0.05


def _rgba(hex_colour: str, alpha: float) -> str:
    """`_rgba` converts hex palette colours to the rgba Plotly fills need."""
    h = hex_colour.lstrip("#")
    return "rgba({}, {}, {}, {})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)), alpha)


#: `PRIOR_AXIS_TOP` must match acquisition.js, which converts pointer positions
#: against it.
PRIOR_AXIS_TOP = 1.25


def slice_span(positions, kind):
    """`slice_span` returns the unit interval for a number and half a step past
    the choices for a categorical. Prior densities are normalized on the same
    span."""
    if kind == "categorical":
        return [min(positions) - 0.5, max(positions) + 0.5]
    return [0.0, 1.0]


def _trace_count(fig: go.Figure) -> int:
    return len(cast(tuple, fig.data))


def acquisition_slice_plots(hp_name, positions, labels, mu, sigma, metric_label,
                            eta=None, higher_is_better=True, kind="continuous",
                            incumbent=None, cloud=None):
    """`acquisition_slice_plots` returns three separate figures so each carries
    its own controls, and they share span, ticks and positions to keep one x
    axis. `meta` holds the data and each figure's trace indices, which the
    browser fills. The function returns `(None, None)` for empty *positions*.
    """
    if not positions:
        return None, None

    blank = [None] * len(positions)
    traces = {"acquisition": {}, "prior": {}, "surrogate": {}}

    # Tick indices, so labels come by position.
    span = slice_span(positions, kind)
    if kind == "categorical":
        shown = range(len(positions))
    else:
        shown = range(0, len(positions), max(1, (len(positions) - 1) // 5))
    ticks = [positions[i] for i in shown]
    text = [sigfigs(labels[i]) for i in shown]

    def _dress(fig, *, title, height, show_x_title=False):
        """Axes are pinned because the prior drag is the only interaction, and a
        *height* of None lets a figure fill its slot.
        """
        fig.update_xaxes(range=span, fixedrange=True, tickvals=ticks,
                         ticktext=text, showticklabels=True)
        if show_x_title:
            fig.update_xaxes(title_text=hp_name)
        fig.update_yaxes(fixedrange=True, title_text=title)
        fig.update_layout(
            dragmode=False, hovermode="x unified", height=height,
            # A top margin keeps the legend above the plot, clear of a sharp
            # prior's peak.
            margin=dict(t=44, b=42 if show_x_title else 24, l=64, r=24),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, borderwidth=0,
                        font=dict(size=11)),
        )
        return fig

    # ── acquisition ──
    acquisition = go.Figure()
    # The sampled envelope shows what the slice misses, since SMAC's maximizer
    # scores a random sample of the whole space. A maximum over a sample is a
    # slack lower bound, and 50,000 samples instead of 2,000 moved it only from
    # 0.17 to 0.34.
    if cloud is not None:
        # The unweighted twin draws first, dotted, so the gap to the weighted
        # curve shows what the prior did. The browser leaves the twin empty
        # until a prior is stated.
        traces["acquisition"]["cloudBare"] = _trace_count(acquisition)
        acquisition.add_trace(go.Scatter(
            x=positions, y=list(blank), mode="lines", connectgaps=False,
            name=_("Best sampled, unweighted"), hoverinfo="skip",
            line=dict(width=1.2, color=MARKER_COLOR, dash="dot")))
        traces["acquisition"]["cloud"] = _trace_count(acquisition)
        acquisition.add_trace(go.Scatter(
            x=positions, y=list(blank), mode="lines", connectgaps=False,
            name=_("Best sampled, weighted"), hoverinfo="skip",
            line=dict(width=1.2, color=MARKER_COLOR), fill="tozeroy",
            fillcolor=_rgba(MARKER_COLOR, 0.10)))

    # The browser scales both curves by one divisor, since only an acquisition
    # function's ranking matters.
    traces["acquisition"]["acquisition"] = _trace_count(acquisition)
    acquisition.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", name=_("Acquisition"),
        hoverinfo="skip", line=dict(width=1.4, color=MARKER_COLOR, dash="dash")))
    traces["acquisition"]["weighted"] = _trace_count(acquisition)
    acquisition.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", name=_("Weighted by the prior"),
        hoverinfo="skip", line=dict(width=2.5, color=PRIOR_COLOR)))
    # Other trials go unmarked because the incumbent is the one measured
    # configuration on the slice curve. The browser fills the marker only while
    # the slice curves show, at an acquisition height.
    traces["acquisition"]["incumbent"] = _trace_count(acquisition)
    acquisition.add_trace(go.Scatter(
        x=[], y=[], mode="markers", name=_("Incumbent"),
        marker=dict(size=9, color=MARKER_COLOR,
                    line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate=_("Incumbent") + "<extra></extra>"))
    # Candidates sit as a rug along the top, since browser-computed heights can
    # disagree with the search's ranking and misplace the "next" marker.
    _RUG_Y = 1.03
    #: The browser converts `_MARKER_PX` to axis units and merges candidates
    #: within one marker width.
    _MARKER_PX = 13
    traces["acquisition"]["candidates"] = _trace_count(acquisition)
    acquisition.add_trace(go.Scatter(
        # Candidates share one trace and one style, since a second style would
        # reintroduce ranking by eye.
        x=[], y=[], mode="markers+text", name=_("Optimizer's Candidate"),
        customdata=[], text=[], textposition="top center",
        textfont=dict(size=10, color=ACCENT_COLOR),
        marker=dict(size=_MARKER_PX, symbol="diamond", color=ACCENT_COLOR,
                    line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="%{customdata[0]}<extra></extra>"))

    _dress(acquisition, title=_("Acquisition"), height=None)
    acquisition.update_yaxes(range=[0, 1.12])
    # The hover label sits on the marker, since "x unified" pushed tall labels
    # off the figure.
    acquisition.update_layout(
        hovermode="closest",
        hoverlabel=dict(align="left", font=dict(size=11),
                        bgcolor="rgba(255, 255, 255, 0.96)",
                        bordercolor=_rgba(ACCENT_COLOR, 0.5)))

    # ── prior ──
    prior = go.Figure()
    traces["prior"]["prior"] = _trace_count(prior)
    prior.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", name=_("Prior"), fill="tozeroy",
        fillcolor=_rgba(PRIOR_COLOR, 0.13), line=dict(width=2, color=PRIOR_COLOR),
        hovertemplate="%{y:.3g}<extra></extra>"))
    # The tabulated prior's control points form the one trace whose x moves when
    # dragged. For a categorical prior the points stay on choice indices.
    traces["prior"]["priorPoints"] = _trace_count(prior)
    prior.add_trace(go.Scatter(
        x=[], y=[], mode="markers", name=_("Control points"), showlegend=False,
        marker=dict(size=9, color=PRIOR_COLOR, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="%{y:.3g}<extra></extra>"))
    _dress(prior, title=_("Prior"), height=None)
    # The prior axis is linear and pinned at `PRIOR_AXIS_TOP`, since the curve
    # is peak-normalized. The headroom keeps a uniform prior's flat line at 1
    # clear of the top edge.
    prior.update_yaxes(type="linear", range=[0, PRIOR_AXIS_TOP], autorange=False)

    # ── surrogate ──
    surrogate = go.Figure()
    # Three nested σ bands show where the mass sits, drawn widest first.
    for k in SIGMA_STEPS:
        surrogate.add_trace(go.Scatter(
            x=positions, y=[m - k * sd for m, sd in zip(mu, sigma)], mode="lines",
            hoverinfo="skip", line=dict(width=0), showlegend=False))
        surrogate.add_trace(go.Scatter(
            x=positions, y=[m + k * sd for m, sd in zip(mu, sigma)], mode="lines",
            hoverinfo="skip", line=dict(width=0), fill="tonexty",
            fillcolor=_rgba(ACCENT_COLOR, _BAND_ALPHA[k]), showlegend=False))
    surrogate.add_trace(go.Scatter(
        x=positions, y=mu, mode="lines", name=_("Predicted"),
        line=dict(width=2.5, color=ACCENT_COLOR),
        hovertemplate="%{y:.4g}<extra></extra>"))

    # The incumbent is the one measured point on this slice.
    if incumbent is not None:
        surrogate.add_trace(go.Scatter(
            x=[incumbent[0]], y=[incumbent[1]], mode="markers", name=_("Incumbent"),
            marker=dict(size=9, color=MARKER_COLOR,
                        line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate=_("Incumbent") + ": %{y:.4g}<extra></extra>"))

    # The implied surrogate gets one faint band. `connectgaps=False` keeps gaps
    # where the prior claims more improvement than any finite mean allows.
    lo = _trace_count(surrogate)
    surrogate.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", hoverinfo="skip",
        line=dict(width=0), connectgaps=False, showlegend=False))
    surrogate.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", hoverinfo="skip",
        line=dict(width=0), fill="tonexty", fillcolor=_rgba(PRIOR_COLOR, _GHOST_ALPHA),
        connectgaps=False, showlegend=False))
    traces["surrogate"]["fictionBand"] = [lo, lo + 1]
    traces["surrogate"]["fiction"] = _trace_count(surrogate)
    surrogate.add_trace(go.Scatter(
        x=positions, y=list(blank), mode="lines", name=_("Implied by the prior"),
        hoverinfo="skip", connectgaps=False,
        line=dict(width=1.6, color=_rgba(PRIOR_COLOR, 0.5), dash="dot")))
    _dress(surrogate, title=metric_label, height=330, show_x_title=True)

    meta = {
        "traces": traces, "sigmas": list(SIGMA_STEPS), "fictionSigma": FICTION_SIGMA,
        "rugY": _RUG_Y, "markerPx": _MARKER_PX,
        # The browser receives the raw sample and computes expected improvement
        # with its curve function.
        "cloud": ({"positions": cloud[0], "mu": cloud[1], "sigma": cloud[2]}
                  if cloud is not None else None),
        "positions": list(positions), "mu": list(mu), "sigma": list(sigma),
        # The acquisition panel uses only the position, and the surrogate panel
        # uses the score.
        "incumbent": list(incumbent) if incumbent is not None else None,
        "eta": eta, "higherIsBetter": bool(higher_is_better),
        "span": span, "kind": kind, "hp": hp_name,
    }
    return {"acquisition": acquisition, "prior": prior, "surrogate": surrogate}, meta


def local_effects_plot(hp_names: list, rows: list):
    """`local_effects_plot` draws every sampled trial's local ablation as a
    beeswarm, one row per hyperparameter, to show spread that averages hide.
    Rows are ordered by spread with the widest on top, points get deterministic
    jitter, and colour follows sign. The function returns None with nothing to
    plot.
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
        # Deterministic jitter keeps the figure stable across reloads.
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
    # Rules between rows separate the jittered points.
    for index in range(1, len(ordered)):
        fig.add_hline(y=index - 0.5,
                      line=dict(color="rgba(140,140,140,0.25)", width=1))
    fig.update_layout(
        xaxis_title=_("Effect vs. default  (left: hurt, right: helped)"),
        yaxis=dict(tickmode="array", tickvals=list(range(len(ordered))), ticktext=ordered),
        margin=dict(t=20, b=40, l=110, r=20), showlegend=False,
        # Each row is a trace holding the selected trial too, and the sampled
        # points carry their trial indices.
        meta={"selection": _selection_meta(
            {i: [row["index"] for row in rows] for i in range(len(ordered))},
            style=OUTLINE)},
    )
    return fig


def trial_duration_plot(result):
    """`trial_duration_plot` tints failed trials, since a duration stays real
    when the score is missing, and returns None with no trials.
    """
    trials = result.trials
    if not trials:
        return None
    # Bars are fills, and failures therefore get a tint.
    fig = go.Figure(go.Bar(
        x=[t.trial for t in trials], y=[t.duration for t in trials],
        marker_color=[FAILURE_FILL if t.failed else MARKER_COLOR for t in trials],
    ))
    fig.update_layout(
        xaxis_title=_("Trial"), yaxis_title=_("Duration (s)"),
        margin=dict(t=20, b=40, l=40, r=20), showlegend=False,
        meta={"selection": _selection_meta({0: range(len(trials))},
                                           style=RECOLOR)},
    )
    return fig


def forecast_plot(times, actual, backtests, target="", series="", labels=False):
    """`forecast_plot` draws the actual series plain and first, then each
    backtest's forecast in the accent colour with its scored points. With
    *labels*, the axis is categorical and forecasts are points only, since a
    line between labels would claim values between them.
    """
    if len(actual) == 0:
        return None
    value = str if labels else float
    shown = "%{y}" if labels else "%{y:.4g}"
    fig = go.Figure(go.Scatter(
        x=list(times), y=[value(v) for v in actual], mode="lines",
        name=series or _("Actual"),
        line=dict(color=MARKER_COLOR, width=1.5, shape="hv" if labels else "linear"),
        hovertemplate="%{x}<br>" + shown + "<extra>" + _("actual") + "</extra>",
    ))
    for number, (when, forecast) in enumerate(backtests, start=1):
        if not len(forecast):
            continue
        fig.add_trace(go.Scatter(
            x=list(when), y=[value(v) for v in forecast],
            mode="markers" if labels else "lines+markers",
            name=_("Backtest %(n)s") % {"n": number},
            line=dict(color=ACCENT_COLOR, width=2, dash="dot"),
            marker=dict(size=8 if labels else 5, color=ACCENT_COLOR),
            hovertemplate="%{x}<br>" + shown + "<extra>" + _("forecast") + "</extra>",
        ))
    fig.update_layout(
        xaxis_title="", yaxis_title=target,
        margin=dict(t=20, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    if labels:
        fig.update_yaxes(type="category", categoryorder="category ascending")
    return fig
