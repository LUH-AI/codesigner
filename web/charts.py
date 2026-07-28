"""Plotly figure builders for the results panels.

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


def performance_figure(result, display_metric, selected_idx=None):
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


def importance_figure(result, display_metric):
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


def duration_figure(result):
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


def gain_per_time_figure(result, display_metric):
    """Return-on-compute: best-so-far (for *display_metric*) ÷ cumulative trial
    time, as a line, with a marker at each trial that set a new incumbent.

    Between wins the best is flat while time grows, so the line descends —
    utility eroding as compute is spent without improvement; an efficient win
    steps it back up. The markers flag every new incumbent, so even a slow win
    that barely moves the line is visible. The first trial is dropped: its
    best÷time is a huge outlier (a full score over a tiny time) that flattens
    the scale. Returns None with fewer than two trials.
    """
    trials = result.trials
    if len(trials) < 2:
        return None
    incumbents = incumbent_scores(result, display_metric)
    elapsed, running = [], 0.0
    for t in trials:
        running += t.duration
        elapsed.append(running)

    def utility(i):
        return incumbents[i] / elapsed[i] if elapsed[i] > 1e-9 else 0.0

    # Plot from the second trial on (drop the out-of-scale first point).
    xs = [trials[i].trial for i in range(1, len(trials))]
    ys = [utility(i) for i in range(1, len(trials))]
    wins_x = [trials[i].trial for i in range(1, len(trials)) if incumbents[i] > incumbents[i - 1]]
    wins_y = [utility(i) for i in range(1, len(trials)) if incumbents[i] > incumbents[i - 1]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Return on compute",
        line=dict(width=2, color=_MARKER_COLOR),
    ))
    if wins_x:
        fig.add_trace(go.Scatter(
            x=wins_x, y=wins_y, mode="markers", name="New incumbent",
            marker=dict(size=11, color=_SELECTED_COLOR, symbol="triangle-up"),
        ))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Best ÷ cumulative s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig
