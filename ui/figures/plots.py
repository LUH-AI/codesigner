"""Plotly chart builders, one per figure that draws a figure.

Each is named for the figure it backs (see `catalog.py`), so
`performance_over_time_plot` fills the "Performance over time" figure. The
other figures — best/selected configuration and trials — are tables, and are
built by their templates from the view's context rather than from here; so is
hyperparameter importance's "table" view, alongside this module's pie/bar.

Pure functions: given an OptimizationResult and a metric, return a
plotly.graph_objects.Figure (or None). No Django, no request state — the view
serializes these with fig.to_json() and the browser renders them.
"""

import plotly.graph_objects as go

_MARKER_COLOR = "#636EFA"
_SELECTED_COLOR = "#EF553B"


def incumbent_scores(result, display_metric):
    """The running best (non-decreasing) score by *display_metric*, per trial."""
    best = float("-inf")
    out = []
    for t in result.trials:
        best = max(best, t.scores[display_metric])
        out.append(best)
    return out


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
    if not trials:
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

    colors = [_MARKER_COLOR] * len(trials)
    sizes = [6] * len(trials)
    if selected_idx is not None and 0 <= selected_idx < len(trials):
        colors[selected_idx] = _SELECTED_COLOR
        sizes[selected_idx] = 13

    improved = [i == 0 or incumbents[i] > incumbents[i - 1] for i in range(len(trials))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=outcome_name,
        marker=dict(size=sizes, color=colors, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=incumbent_ys, mode="lines", name="Incumbent",
        line=dict(width=2, shape="hv"),
    ))
    fig.add_trace(go.Scatter(
        x=[xs[i] for i in range(len(trials)) if improved[i]],
        y=[incumbent_ys[i] for i in range(len(trials)) if improved[i]],
        mode="markers", name="New incumbent",
        marker=dict(size=9, color=_SELECTED_COLOR, symbol="diamond"),
    ))
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=y_title, yaxis_type=y_type,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def hyperparameter_importance_plot(importance: dict, rendering: str = "pie"):
    """Pie or (vertical) bar of a per-hyperparameter importance dict — shared
    by every HyperSHAP game this app surfaces (tunability, sensitivity,
    mistunability all produce the same shape: {hp: normalized weight}), the
    caller picks which dict to hand in. A third rendering, "table", is
    rendered directly by the template from `panels`, not from here.

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
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=_MARKER_COLOR))
        fig.update_layout(yaxis_title="Importance",
                          margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        return fig
    fig = go.Figure(go.Pie(
        labels=list(importance.keys()), values=list(importance.values()),
        hole=0.35, textinfo="label+percent",
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
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
        marker_color=_MARKER_COLOR,
    ))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Duration (s)",
        margin=dict(t=20, b=40, l=40, r=20), showlegend=False,
    )
    return fig
