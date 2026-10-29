# Immediate `gettext`, since `to_json` cannot serialize a lazy proxy.
from django.utils.translation import gettext as _
import plotly.graph_objects as go

from ...formatting import sigfigs
from .style import MARKER_COLOR, NEGATIVE_COLOR, TEXT_FONT, UNIFORM_TEXT, signed


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
        insidetextorientation="horizontal", textfont=TEXT_FONT,
        # `hole=0` is explicit, since `Plotly.react` treats an absent attribute
        # more weakly than a default.
        hole=0,
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
                      **UNIFORM_TEXT)
    return fig


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
        textfont=TEXT_FONT, sort=False, direction="clockwise", hole=0,
        marker=dict(colors=colours, line=dict(width=1, color="#FFFFFF")),
        hovertext=hovers, hoverinfo="text",
    ))
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False,
                      **UNIFORM_TEXT)
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
        text=[""] + [signed(v) for v in values] + [""],
        textposition="outside",
        increasing=dict(marker=dict(color=MARKER_COLOR)),
        decreasing=dict(marker=dict(color=NEGATIVE_COLOR)),
        totals=dict(marker=dict(color="#8C9196")),
        connector=dict(line=dict(color="rgba(140,140,140,0.5)", width=1)),
    ))
    fig.update_layout(yaxis_title=_("vs. default"),
                      margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
    return fig
