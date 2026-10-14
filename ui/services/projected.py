"""Priors drawn in the configuration cube, between the page and
`core.projected_prior`.

The page picks points and moves sliders; everything a belief *is* is worked out
here — the record it becomes, its weight over the cube's shared sample, and the
shapes the cube draws for it — so the browser draws and multiplies but computes
none of the math, and what it shows is what SMAC will weigh by.

Points arrive in the units the cube shows. A projection's are its components;
HP Space's are the hyperparameters' own values, which go onto the vectorized
axis every prior is stated on (`ConfigSpace`'s own `to_vector`) and come back
off it only to be drawn.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
from django.utils.translation import gettext as _

from core import projected_prior as pp

#: How far a default width reaches, as a share of the view's extent.
_DEFAULT_WIDTH = 0.1

#: Vertices on a drawn circle.
_CIRCLE = 64

_VIEW_NAMES = {"axes": "HP Space", "pca": "PCA", "pls": "PLS"}


def _numeric(hp) -> bool:
    return (hasattr(hp, "lower") and hasattr(hp, "upper")
            and getattr(hp, "choices", None) is None)


def _to_vector(hp, value) -> float:
    return float(hp.to_vector(np.array([float(value)]))[0])


def _display(hp, vector):
    """A vectorized position on *hp*'s axis as the cube draws it: the inverse
    of `to_vector`, without rounding an integer's position to a whole value, so
    a line stays a line."""
    v = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    lower, upper = float(hp.lower), float(hp.upper)
    if getattr(hp, "log", False) and lower > 0:
        return lower * (upper / lower) ** v
    return lower + v * (upper - lower)


def sample_vectors(sample) -> np.ndarray:
    """The shared sample's vectorized rows, NaN where a hyperparameter is
    inactive."""
    return np.array([[np.nan if v is None else v for v in row] for row in sample["vectors"]],
                    dtype=float)


def label(record) -> str:
    """"PLS · 2D · 2 points", or "HP Space (max_depth, n_estimators) · 1 point"."""
    view = _VIEW_NAMES.get(record["view"], record["view"])
    if record["view"] == "axes":
        view += " (" + ", ".join(record["hyperparameters"]) + ")"
    else:
        view += f" · {pp.dimensions(record)}D"
    points = len(record["points"])
    return f"{view} · {points} point" + ("s" if points != 1 else "")


def draft_record(config_space, sample, draft, metric):
    """What the page has drawn, as a record `core.projected_prior` evaluates —
    and the shared sample's coordinates in its view, which the caller weighs.

    Missing parameters take defaults: positions halfway, spreads covering the
    segment or triangle, and a width a tenth of the view's extent. Raises
    `PriorError` for what cannot be a prior."""
    view = draft.get("view")
    points = [[float(v) for v in point] for point in draft.get("points") or []]
    if not points:
        raise pp.PriorError("Pick a point first.")
    dims = len(points[0])
    if view == "axes":
        names = list(draft.get("hyperparameters") or [])[:dims]
        if len(names) != dims or any(n not in config_space for n in names):
            raise pp.PriorError("The axes are not hyperparameters of this search space.")
        if not all(_numeric(config_space[n]) for n in names):
            raise pp.PriorError("A prior can only be drawn on numeric axes.")
        record = {"view": "axes", "hyperparameters": names,
                  "points": [[_to_vector(config_space[n], v) for n, v in zip(names, point)]
                             for point in points]}
    elif view in ("pca", "pls"):
        projection = (sample.get("projections") or {}).get(view)
        if not projection:
            raise pp.PriorError("This projection could not be computed.")
        linear = projection["map"]
        record = {"view": view, "points": points,
                  "map": {"columns": linear["columns"], "matrix": linear["matrix"][:dims],
                          "offset": linear["offset"][:dims]}}
    else:
        raise pp.PriorError(f"No such view: {view!r}.")

    pp.frame(record["points"])
    _refuse_outside(record, sample)
    Z = pp.view_coordinates(record, config_space, sample_vectors(sample))
    along = len(points) - 1
    params = draft.get("params") or {}
    extent = float(np.nanmean(np.nanmax(Z, axis=0) - np.nanmin(Z, axis=0))) if len(Z) else 1.0
    record["params"] = {
        "mean": [float(v) for v in (params.get("mean") or [0.5] * along)][:along],
        "spread": [max(float(v), 1e-6) for v in (params.get("spread") or [0.5] * along)][:along],
        "width": max(float(params.get("width") or round(_DEFAULT_WIDTH * extent, 4)), 1e-6),
    }
    record["kind"] = "normal"
    record["metric"] = metric
    record["label"] = label(record)
    pp.check(record, config_space)
    return record, Z


def with_params(record, params) -> dict:
    """*record* with its position, spread and width changed — what editing a
    stated prior may change; anything not given stays as it was."""
    along = len(record["points"]) - 1
    old = record.get("params") or {}
    mean = params.get("mean") or old.get("mean") or [0.5] * along
    spread = params.get("spread") or old.get("spread") or [0.5] * along
    width = params.get("width") or old.get("width") or 0.1
    return dict(record, params={"mean": [float(v) for v in mean][:along],
                                "spread": [max(float(v), 1e-6) for v in spread][:along],
                                "width": max(float(width), 1e-6)})


def outside(record, sample):
    """The first of *record*'s points that is no configuration at all — off a
    hyperparameter's range in HP Space, beyond the image of the search space in
    a projection (the outline the cube draws) — counted from 0, or None."""
    points = np.asarray(record["points"], dtype=float)
    if record["view"] == "axes":
        for i, point in enumerate(points):
            if np.any(point < -1e-9) or np.any(point > 1 + 1e-9):
                return i
        return None
    dims = points.shape[1]
    projection = (sample.get("projections") or {}).get(record["view"]) or {}
    outline = (projection.get("outlines") or {}).get(str(dims))
    if outline is None:
        return None
    if dims == 1:
        lo, hi = float(outline[0]), float(outline[1])
        slack = 1e-6 * max(hi - lo, 1e-12)
        for i, point in enumerate(points):
            if point[0] < lo - slack or point[0] > hi + slack:
                return i
        return None
    from scipy.spatial import ConvexHull

    vertices = (np.asarray(outline, dtype=float) if dims == 2
                else np.column_stack([outline["x"], outline["y"], outline["z"]]))
    hull = ConvexHull(vertices, qhull_options="QJ")
    slack = 1e-6 * float(np.ptp(vertices, axis=0).max() or 1.0)
    for i, point in enumerate(points):
        if np.any(hull.equations[:, :-1] @ point + hull.equations[:, -1] > slack):
            return i
    return None


def _refuse_outside(record, sample):
    at = outside(record, sample)
    if at is not None:
        error = pp.PriorError(_("Point %(n)s lies outside the search space.") % {"n": at + 1})
        error.point = at
        raise error


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def display_points(record, config_space):
    """*record*'s points in the units the cube shows."""
    if record["view"] != "axes":
        return record["points"]
    return [[float(_display(config_space[n], v)) for n, v in zip(record["hyperparameters"], point)]
            for point in record["points"]]


def summary(record, config_space) -> dict:
    """A stated prior as its card lists it."""
    return {"id": record.get("id"), "label": record.get("label"), "view": record["view"],
            "hyperparameters": record.get("hyperparameters") or [],
            "dims": pp.dimensions(record),
            "points": display_points(record, config_space) if config_space is not None
            else record["points"],
            "params": record.get("params") or {}, "decay": record.get("decay"),
            "delay_decay": bool(record.get("delay_decay")),
            "at_trial": record.get("at_trial"), "metric": record.get("metric"),
            "profile": pp.profile(record)}


def rounded(values, figures=4):
    return [float(f"{v:.{figures}g}") if np.isfinite(v) else 0.0 for v in np.asarray(values, float)]


def stated_weights(records, exponents, config_space, X):
    """Every stated projected prior's weight at the vectorized rows *X*, at the
    exponent its decay has reached — floored before the exponent, as SMAC's
    weight layer does — multiplied together. None with nothing stated."""
    if not records:
        return None
    out = np.ones(len(X))
    for record, exponent in zip(records, exponents):
        try:
            out = out * np.power(pp.weight(record, config_space, X) + 1e-12, exponent)
        except (pp.PriorError, KeyError, ValueError):
            continue
    return out


# ── what the cube draws ─────────────────────────────────────────────────────

def _clip_line(origin, direction, lo, hi):
    """The part of the line `origin + s·direction` inside the box, or None."""
    s_min, s_max = -math.inf, math.inf
    for o, e, a, b in zip(origin, direction, lo, hi):
        if abs(e) < 1e-12:
            if not a <= o <= b:
                return None
            continue
        s1, s2 = (a - o) / e, (b - o) / e
        s_min, s_max = max(s_min, min(s1, s2)), min(s_max, max(s1, s2))
    if s_min > s_max:
        return None
    return [list(origin + s_min * direction), list(origin + s_max * direction)]


def _plane_patch(centre, e1, e2, lo, hi):
    """Where the plane through *centre* spanned by *e1*, *e2* crosses the box:
    a polygon, as a triangle fan."""
    normal = np.cross(e1, e2)
    corners = [np.array([lo[0] if i & 1 == 0 else hi[0], lo[1] if i & 2 == 0 else hi[1],
                         lo[2] if i & 4 == 0 else hi[2]]) for i in range(8)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    points = []
    for a, b in edges:
        da, db = normal @ (corners[a] - centre), normal @ (corners[b] - centre)
        if da == db or da * db > 0:
            continue
        points.append(corners[a] + (da / (da - db)) * (corners[b] - corners[a]))
    if len(points) < 3:
        return None
    middle = np.mean(points, axis=0)
    points.sort(key=lambda p: math.atan2((p - middle) @ e2, (p - middle) @ e1))
    xs, ys, zs = zip(*[list(p) for p in [middle] + points])
    n = len(points)
    return {"x": list(xs), "y": list(ys), "z": list(zs),
            "i": [0] * n, "j": [1 + i for i in range(n)], "k": [1 + (i + 1) % n for i in range(n)]}


def geometry(record, Z, config_space) -> dict:
    """The shapes the cube draws for *record*, in the units it shows: the
    points, the peak (`mode`), the line or plane through the points cut to the
    space, the width as a band either side (a circle around a single point in
    two dimensions; marks either side of it in one), and one spread either side
    of the peak along each direction (`ticks`)."""
    origin, directions, scales = pp.frame(record["points"])
    mean, spread, width = pp.params_of(record, len(directions))
    d = len(origin)
    if record["view"] == "axes":
        lo, hi = np.zeros(d), np.ones(d)
    else:
        lo, hi = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    mode = origin + ((mean * scales) @ directions if len(directions) else 0)

    lines, band, ticks, marks, mesh = [], [], [], [], None
    if len(directions) >= 1:
        line = _clip_line(origin, directions[0], lo, hi)
        if line:
            lines.append(line)
        for j, direction in enumerate(directions):
            for sign in (-1, 1):
                ticks.append(list(mode + sign * spread[j] * scales[j] * direction))
    if d == 2 and len(directions) == 1:
        normal = np.array([-directions[0][1], directions[0][0]])
        for sign in (-1, 1):
            edge = _clip_line(origin + sign * width * normal, directions[0], lo, hi)
            if edge:
                band.append(edge)
    if d == 2 and len(directions) == 0:
        band.append([list(origin + width * np.array([math.cos(t), math.sin(t)]))
                     for t in np.linspace(0, 2 * math.pi, _CIRCLE)])
    if d == 1:
        marks = [float(origin[0] - width), float(origin[0] + width)]
    if d == 3 and len(directions) == 2:
        mesh = _plane_patch(mode, directions[0], directions[1], lo, hi)
        lines.append([list(p) for p in record["points"]] + [list(record["points"][0])])

    def shown(rows):
        rows = np.asarray(rows, dtype=float)
        if record["view"] != "axes" or not rows.size:
            return rows.tolist()
        out = rows.copy()
        for k, name in enumerate(record["hyperparameters"]):
            out[..., k] = _display(config_space[name], rows[..., k])
        return out.tolist()

    if mesh is not None:
        stacked = shown(np.column_stack([mesh["x"], mesh["y"], mesh["z"]]))
        mesh.update(x=[p[0] for p in stacked], y=[p[1] for p in stacked], z=[p[2] for p in stacked])
    if record["view"] == "axes" and marks:
        marks = [float(_display(config_space[record["hyperparameters"][0]], m)) for m in marks]
    return {"points": shown(record["points"]), "mode": shown([mode])[0],
            "lines": [shown(line) for line in lines], "band": [shown(edge) for edge in band],
            "ticks": shown(ticks), "marks": marks, "mesh": mesh}
