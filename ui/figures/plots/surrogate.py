from typing import cast

# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _
import plotly.graph_objects as go

from ...formatting import sigfigs
from .style import (
    ACCENT_COLOR, MARKER_COLOR, NEGATIVE_COLOR, OUTLINE, selection_meta, signed,
)


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
            text=[f"Trial {row['trial']}<br>{name} {signed(v)}"
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
        meta={"selection": selection_meta(
            {i: [row["index"] for row in rows] for i in range(len(ordered))},
            style=OUTLINE)},
    )
    return fig
