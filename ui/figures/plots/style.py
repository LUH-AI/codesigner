from typing import Any

from ...formatting import sigfigs


# Each palette colour means one thing on every figure, by convention only.
# `MARKER_COLOR` marks trials and positive signed values, `ACCENT_COLOR` marks
# the best so far, `NEGATIVE_COLOR` marks negative signed values, and
# `SELECTION_COLOR` marks the selected trial alone. `MARKER_COLOR` and
# `SELECTION_COLOR` are public because the page restyles the selection itself
# (see `selection_colors` in ui/views/dashboard.py).
MARKER_COLOR = "#636EFA"
#: `SELECTION_COLOR` is the soft coral that `#EF553B` became when faded over
#: Plotly's background, painted at full opacity on every figure so the selection
#: looks the same everywhere. `--selected` in app.css holds the same value.
SELECTION_COLOR = "#EC8273"
ACCENT_COLOR = "#10B981"
NEGATIVE_COLOR = "#FF2B2B"
#: A proposed configuration from the cube, which nobody has run, gets its own
#: colour so it cannot pass for a trial. `--proposal` in app.css has the same
#: value.
PROPOSAL_COLOR = "#8B5CF6"

#: The uncertainty field uses a violet ramp, a hue the palette leaves unused,
#: since green means the best so far and blue already encodes the metric on the
#: same figure. Both ends are translucent so the field stays scenery, and the
#: ramp rises with uncertainty.
UNCERTAINTY_SCALE = [
    [0.0, "rgba(124, 58, 237, 0.02)"],
    [1.0, "rgba(124, 58, 237, 0.42)"],
]

#: A failed trial is drawn as an `x-thin` cross in `NEGATIVE_COLOR`, and
#: `marker.line` gives the unfilled glyph its visibility and weight. A selected
#: failure keeps its cross and takes the selection colour.
FAILURE_SYMBOL = "x-thin"
FAILURE_LINE_WIDTH = 3
#: Thinner on the cube and the projection, whose ringed markers would make a
#: full-weight cross the loudest mark.
FAILURE_LINE_WIDTH_RINGED = 2
#: Large enough to read as a cross.
FAILURE_SIZE = 11
#: Tint for failures drawn as fills, such as duration bars.
FAILURE_FILL = "rgba(255, 43, 43, 0.45)"


def translucent(color: str, alpha: float = 0.7) -> str:
    """`translucent` lets a trace fade individual points and keep a failure's
    cross at full strength.
    """
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def failure_marks(trials, *, size, symbol="circle", line_width=1,
                   line_color="#FFFFFF", selected_idx=None,
                   failure_width=FAILURE_LINE_WIDTH):
    """`failure_marks` returns marker keys that mark failed trials, or `{}`
    when none failed. It styles the existing trace instead of adding one, since
    a failed trial must stay selectable and `selection_meta` maps trace
    positions to trials one for one.
    """
    failed = [t.failed for t in trials]
    if not any(failed):
        return {}

    sizes = list(size) if isinstance(size, (list, tuple)) else [size] * len(trials)
    widths = list(line_width) if isinstance(line_width, (list, tuple)) else [line_width] * len(trials)
    colors = list(line_color) if isinstance(line_color, (list, tuple)) else [line_color] * len(trials)

    for i, bad in enumerate(failed):
        if not bad:
            continue
        sizes[i] = max(sizes[i], FAILURE_SIZE)
        widths[i] = failure_width
        colors[i] = (SELECTION_COLOR if i == selected_idx else NEGATIVE_COLOR)

    return {
        "symbol": [FAILURE_SYMBOL if bad else symbol for bad in failed],
        "size": sizes,
        "line": {"width": widths, "color": colors},
    }


#: Selection styles; the figure's colour meaning decides which applies.
RECOLOR = "recolor"   # free colour: paint the selected point
OUTLINE = "outline"   # colour carries meaning: ring it
LINE = "line"         # polyline per trial (parallel coordinates)
OVERLAY = "overlay"   # selected point redrawn on top


def signed(value) -> str:
    """`signed` always writes the sign, since helped or hurt is the reading
    here.
    """
    return f"{'+' if value >= 0 else '-'}{sigfigs(abs(value))}"


def selection_meta(by_trace: dict, *, style, segment=1, **extra) -> dict:
    """`selection_meta` tells the page's selection script which trial each
    clicked point is, via `layout.meta["selection"]`. *by_trace* maps a trace
    index to positions in `result.trials`, because traces outside the map (an
    incumbent line, a colour-bar carrier) carry no trials and local effects
    draws a sample. *segment* counts points per trial, 1 except in parallel
    coordinates, so the trial at point *p* of trace *t* is `by_trace[t][p //
    segment]`.
    """
    return dict(trials={str(k): list(v) for k, v in by_trace.items()},
                style=style, segment=segment, **extra)


#: `textfont` fixes one label size for every wedge, and `uniformtext` hides
#: labels that no longer fit, leaving them in the hover. `minsize` equals the
#: font size, which keeps Plotly from scaling labels again.
TEXT_SIZE = 12
TEXT_FONT = {"size": TEXT_SIZE}
UNIFORM_TEXT: dict[str, Any] = {"uniformtext": {"minsize": TEXT_SIZE, "mode": "hide"}}
