"""Plotly chart builders, one per figure that draws a figure.

Each is named for the figure it backs (see `catalog.py`), so
`incumbent_performance_plot` fills the "Performance of Incumbent" figure. The
other figures — best/selected configuration and trials — are tables, and are
built by their templates from the view's context rather than from here.

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


def incumbent_performance_plot(result, display_metric, selected_idx=None):
    """Scatter of each trial's score with the running-best line overlaid.

    The point at *selected_idx* (default: none) is enlarged and recolored, the
    same highlight the click-to-select feature will drive later.
    """
    trials = result.trials
    scores = [t.scores[display_metric] for t in trials]
    incumbents = incumbent_scores(result, display_metric)

    colors = [_MARKER_COLOR] * len(trials)
    sizes = [6] * len(trials)
    if selected_idx is not None and 0 <= selected_idx < len(trials):
        colors[selected_idx] = _SELECTED_COLOR
        sizes[selected_idx] = 13

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t.trial for t in trials], y=scores,
        mode="markers", name="Trial score",
        marker=dict(size=sizes, color=colors, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=[t.trial for t in trials], y=incumbents,
        mode="lines", name="Incumbent", line=dict(width=2),
    ))
    fig.update_layout(
        xaxis_title="Trial",
        yaxis_title=display_metric.capitalize(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def hyperparameter_importance_plot(result, display_metric):
    """Donut of hyperparameter importance for *display_metric*.

    Returns None when no importance was computed for the metric (the caller
    shows an explanatory message instead).
    """
    imp = result.hyperparameter_importance.get(display_metric, {})
    if not imp:
        return None
    fig = go.Figure(go.Pie(
        labels=list(imp.keys()), values=list(imp.values()),
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


def error_over_time_plot(result, display_metric):
    """Remaining error (1 − best-so-far) against cumulative trial time — an
    "anytime performance" curve: how the error comes down as compute is spent.

    Unlike the incumbent line (error vs *trial number*), the x-axis here is
    seconds of compute, so the WIDTH of each flat run is time spent without
    improvement (expensive dry spells stretch wide) and each drop is a win.
    Log y so near-optimal gains still read large; markers flag new incumbents.
    Returns None with no trials.
    """
    trials = result.trials
    if not trials:
        return None
    incumbents = incumbent_scores(result, display_metric)
    elapsed, running = [], 0.0
    for t in trials:
        running += t.duration
        elapsed.append(running)
    regret = [max(1e-3, 1.0 - incumbents[i]) for i in range(len(trials))]  # floor so log never hits 0
    win = [i == 0 or incumbents[i] > incumbents[i - 1] for i in range(len(trials))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elapsed, y=regret, mode="lines", name="Remaining error",
        line=dict(width=2, shape="hv", color=_MARKER_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=[elapsed[i] for i in range(len(trials)) if win[i]],
        y=[regret[i] for i in range(len(trials)) if win[i]],
        mode="markers", name="New incumbent",
        marker=dict(size=10, color=_SELECTED_COLOR),
    ))
    fig.update_layout(
        xaxis_title="time (s)", yaxis_title="Error", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig
