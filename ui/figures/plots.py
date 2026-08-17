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


def hyperparameter_ablation_plot(ablation: dict):
    """Diverging bar of one trial's HyperSHAP ablation values against the
    config space's default — the "Local (selected trial)" view.

    Unlike the three global games' pie/bar (unsigned shares of a whole),
    these values are signed: a positive bar means that hyperparameter's value
    in this trial beat the default, negative means it lost to it. Colored by
    sign for exactly that reason; ranked by magnitude, like the other views,
    since "which mattered most" is still the first question even when some
    answers are negative.

    Returns None when *ablation* is empty (not enough trials, or the game
    failed — see BaseOptimizer.compute_hp_ablation). No "table"/"pie" — a
    share-of-a-whole framing does not apply to signed values, so this is the
    one view this game gets.
    """
    if not ablation:
        return None
    names, values = zip(*sorted(ablation.items(), key=lambda kv: abs(kv[1]), reverse=True))
    colors = [_MARKER_COLOR if v >= 0 else _SELECTED_COLOR for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    fig.update_layout(yaxis_title="vs. default",
                      margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
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
        z=z, x=params, y=params, colorscale="RdBu", zmid=0,
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
    colors = [_MARKER_COLOR if v >= 0 else _SELECTED_COLOR for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    fig.update_layout(yaxis_title="Interaction strength",
                      margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
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
