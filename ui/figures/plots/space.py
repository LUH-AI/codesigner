import math

# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _
import plotly.graph_objects as go

from core.projection import fit, log_hyperparameters

from ...formatting import sigfigs
from .style import (
    FAILURE_LINE_WIDTH_RINGED, LINE, OVERLAY, SELECTION_COLOR, failure_marks,
    selection_meta,
)


#: Score figures use one blue ramp by intensity, since a score has a good end
#: and a bad end with no meaningful middle.
_INTENSITY_SCALE = [
    [0.00, "#EEF1FE"],
    [0.25, "#C3CBFA"],
    [0.50, "#93A0F4"],
    [0.75, "#6373EC"],
    [1.00, "#2B3AA8"],
]


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
    cube_marker.update(failure_marks(trials, size=8,
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
              "selection": selection_meta({0: range(len(trials))},
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
    marker.update(failure_marks(trials, size=8,
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
              "selection": selection_meta({0: range(len(trials))},
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
    `selection_meta`'s `segment` map a point index to a trial.
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
        meta={"selection": selection_meta(
            {trace: [trial] for trace, trial in enumerate(by_score)},
            style=LINE, segment=_densified_length(len(dimensions)),
            # The highlight index counts drawn traces, since failures add none.
            highlight=len(by_score) + 1)},
    )
    return fig
