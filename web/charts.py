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


def _relative_error_reduction(incumbents, i):
    """Fraction of the remaining error (1 − best) that trial i eliminated.

    e.g. 0.65→0.70 cuts error 0.35→0.30 = 14%; 0.95→0.98 cuts 0.05→0.02 = 60%.
    Zero (or negative, clamped) when the trial didn't improve the incumbent.
    """
    prev_err = 1.0 - incumbents[i - 1]
    if prev_err <= 1e-9:
        return 0.0
    return max(0.0, (incumbents[i] - incumbents[i - 1]) / prev_err)


def error_reduction_spikes_figure(result, display_metric):
    """(A) Bars: the fraction of remaining error a trial cut ÷ its duration, on
    a log y-axis. Each win is a spike sized by significance — a 0.65→0.70 jump
    and a 0.95→0.98 jump both stand out. First trial dropped; None with <2
    trials or no post-first improvement."""
    trials = result.trials
    if len(trials) < 2:
        return None
    incumbents = incumbent_scores(result, display_metric)
    xs, ys = [], []
    for i in range(1, len(trials)):
        red = _relative_error_reduction(incumbents, i)
        dur = trials[i].duration
        xs.append(trials[i].trial)
        ys.append(red / dur if dur > 1e-9 else 0.0)
    if not any(v > 0 for v in ys):
        return None
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=_MARKER_COLOR))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Error cut / s", yaxis_type="log",
        margin=dict(t=20, b=40, l=40, r=20), showlegend=False,
    )
    return fig


def regret_convergence_figure(result, display_metric):
    """(B) Remaining error (1 − best-so-far) on a log y-axis over trials — the
    standard convergence view: a descending staircase where each win is a drop
    sized by significance (near-optimal drops tower); markers flag wins. None
    with no trials."""
    trials = result.trials
    if not trials:
        return None
    incumbents = incumbent_scores(result, display_metric)
    regret = [max(1e-3, 1.0 - incumbents[i]) for i in range(len(trials))]  # floor so log never hits 0
    xs = [t.trial for t in trials]
    win = [i == 0 or incumbents[i] > incumbents[i - 1] for i in range(len(trials))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=regret, mode="lines", name="Remaining error",
        line=dict(width=2, shape="hv", color=_MARKER_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=[trials[i].trial for i in range(len(trials)) if win[i]],
        y=[regret[i] for i in range(len(trials)) if win[i]],
        mode="markers", name="New incumbent",
        marker=dict(size=10, color=_SELECTED_COLOR),
    ))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Remaining error (1 − best)", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def return_on_compute_figure(result, display_metric):
    """(C) Declining best-so-far ÷ cumulative time line (utility eroding as
    compute is spent without gains), with a new-incumbent marker whose size
    grows with the error reduction the win achieved. First trial dropped; None
    with <2 trials."""
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

    xs = [trials[i].trial for i in range(1, len(trials))]
    ys = [utility(i) for i in range(1, len(trials))]
    wx, wy, wsize = [], [], []
    for i in range(1, len(trials)):
        if incumbents[i] > incumbents[i - 1]:
            wx.append(trials[i].trial)
            wy.append(utility(i))
            wsize.append(8 + 40 * _relative_error_reduction(incumbents, i))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Return on compute",
        line=dict(width=2, color=_MARKER_COLOR),
    ))
    if wx:
        fig.add_trace(go.Scatter(
            x=wx, y=wy, mode="markers", name="New incumbent",
            marker=dict(size=wsize, color=_SELECTED_COLOR, symbol="triangle-up"),
        ))
    fig.update_layout(
        xaxis_title="Trial", yaxis_title="Best ÷ cumulative s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig
