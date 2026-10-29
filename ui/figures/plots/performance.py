from typing import cast

# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _
import plotly.graph_objects as go

from core.metrics import to_error

from .style import (
    ACCENT_COLOR, FAILURE_FILL, MARKER_COLOR, RECOLOR, SELECTION_COLOR, failure_marks,
    selection_meta, translucent,
)


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
    fills = [c if i == selected_idx else translucent(c)
             for i, c in enumerate(colors)]
    marker = dict(size=sizes, color=fills)
    marker.update(failure_marks(trials, size=sizes, selected_idx=selected_idx))

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
        meta={"selection": selection_meta({0: range(len(trials))},
                                           style=RECOLOR)},
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
        meta={"selection": selection_meta({0: range(len(trials))},
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
