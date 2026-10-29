# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _
import plotly.graph_objects as go

from ...formatting import sigfigs
from .style import MARKER_COLOR, NEGATIVE_COLOR, signed


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
        label = " × ".join(members) + f"<br>{signed(value)}"
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
