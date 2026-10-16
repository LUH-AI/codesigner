"""Priors stated in a view of the search space rather than on one axis of it.

The configuration cube shows the space through a view: some hyperparameters'
own axes (HP Space), or the leading components of a PCA or PLS projection. A
belief drawn there is a Gaussian in that view's coordinates around one, two or
three points:

- one point, a bump around it;
- two, a ridge along the line through them;
- three, a slab around the plane through them.

Whatever the shape, it is one function of a whole configuration, which is the
point of stating it here: a ridge running diagonally across two hyperparameters
says they go together, and no pair of beliefs about each on its own can say
that. So it reaches SMAC as one prior over the whole space (`ProjectedPrior`),
multiplied with the per-hyperparameter beliefs (`CombinedPrior`).

A stated prior is a plain record, so it can be stored, exported and replayed:

    {"view": "axes" | "pca" | "pls",
     "hyperparameters": [...],        # HP Space: the axes, in order
     "map": {"columns", "matrix", "offset"},  # PCA/PLS: the projection, frozen
     "points": [[...], ...],          # view coordinates, one row per point
     "params": {"mean": [...], "spread": [...], "width": w}}

The projection is frozen into the record (`Fitted.linear_map`, cut to the
components the view showed) because PLS moves as trials arrive, and a belief
has to go on meaning what it meant when it was stated.

The math, for points p1..pk in d view dimensions:

- the frame has an origin at p1 and orthonormal directions e1 = p2 − p1 and e2
  = the part of p3 − p1 at right angles to e1;
- positions along them are in natural units: t1 is 0 at p1 and 1 at p2, and t2
  is 0 on the line through p1 and p2 and 1 at p3's distance from it;
- the residual r is what the frame does not reach, and `width` is the Gaussian
  across it;
- the density, unnormalised with a peak of 1, is
  exp(−½ [Σ ((tj − meanj) / spreadj)² + |r|² / width²]).

In HP Space with one point, that is exactly a product of Normals on the
hyperparameters' vectorized axes — the prior every hyperparameter can already
be given on its own.

Drawing from it (`sample_from`), which SMAC's maximizer does for candidates
and its prior judgement does to compare a belief with the incumbent:

- in HP Space the view's coordinates are the hyperparameters' own, so the
  Gaussian is drawn on its axes directly, cut to the space, and the rest of
  the configuration comes from the space or the per-hyperparameter prior;
- in PCA or PLS many configurations land on each place in the view, so there
  is nothing to read a configuration back from. Configurations are aimed at
  the belief instead: a place drawn from the Gaussian, and a random
  configuration's encoding moved the shortest way to one that projects there.
  Draws around those, and some from the whole space, are then weighed by the
  belief over how likely the drawing made them, so what comes out is the
  belief however well or badly the aiming went.

Django-free, like the rest of `core/`.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

import numpy as np

from .projection import _Column, encode_vectors

VIEWS = ("axes", "pca", "pls")

#: The smallest spread or width a prior may have, in its own units. Below it
#: the bump is narrower than anything the search can resolve, and dividing by
#: it is dividing by zero.
_MIN_SCALE = 1e-6

#: How many configurations are weighed for each one handed back when a belief
#: can only be drawn from by weighing (PCA, PLS). Few, because the draws are
#: aimed at it.
_POOL = 4

#: The share of those drawn from the whole space rather than around the
#: belief: what keeps the weighing correct where the aimed draws miss part of
#: it.
_UNIFORM_SHARE = 0.25

#: At most how many configurations are aimed at the belief to draw around.
_CENTRES = 256

#: How often a configuration is moved towards its drawn place in a PCA or
#: PLS view: reading one back keeps it inside the space, so it lands short,
#: and moving it again lets the values that are free make up the rest.
_LIFTS = 6

#: How often a draw that fell outside the space, or broke one of its
#: conditions, is drawn again before the nearest place inside is taken.
_ROUNDS = 8


class PriorError(ValueError):
    """A projected prior that does not describe a shape: points that coincide,
    or more points than the view has dimensions."""

    #: Which point the trouble is with, counted from 0, when it is one point.
    point: Optional[int] = None


def dimensions(record) -> int:
    """How many view dimensions *record* was stated in."""
    if record["view"] == "axes":
        return len(record["hyperparameters"])
    return len(record["map"]["offset"])


def frame(points):
    """The frame *points* span: `(origin, directions, scales)`.

    `directions` are orthonormal rows, one per point after the first; `scales`
    are the lengths that make a position along each read 0 to 1 — the distance
    from p1 to p2, then p3's distance from the line through them. Points that
    coincide, or a third point on the line through the other two, span less
    than they claim, and are refused.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not len(points):
        raise PriorError("A prior needs at least one point.")
    if len(points) > points.shape[1]:
        raise PriorError("More points than the view has dimensions.")
    origin = points[0]
    directions, scales = [], []
    for point in points[1:]:
        offset = point - origin
        for direction in directions:
            offset = offset - (offset @ direction) * direction
        length = float(np.linalg.norm(offset))
        if length < 1e-12:
            raise PriorError("The points coincide or lie on one line.")
        directions.append(offset / length)
        scales.append(length)
    return origin, np.array(directions).reshape(len(directions), points.shape[1]), np.array(scales)


def params_of(record, along: int):
    """*record*'s parameters as arrays — positions and spreads along each of
    its *along* directions, and the width across — each scale kept above
    `_MIN_SCALE`."""
    params = record.get("params") or {}
    mean = np.asarray(params.get("mean") or [0.5] * along, dtype=float)[:along]
    spread = np.maximum(np.asarray(params.get("spread") or [0.5] * along, dtype=float)[:along],
                        _MIN_SCALE)
    width = max(float(params.get("width", 0.1)), _MIN_SCALE)
    return mean, spread, width


def density(Z, record) -> np.ndarray:
    """The prior's density at view coordinates *Z* (one row per configuration),
    unnormalised and peaking at 1. A NaN coordinate — a hyperparameter the
    configuration leaves inactive, on an axis of HP Space — takes the prior's
    own mode there, so a belief about it neither helps nor hurts."""
    Z = np.asarray(Z, dtype=float)
    origin, directions, scales = frame(record["points"])
    mean, spread, width = params_of(record, len(directions))
    if np.isnan(Z).any():
        mode = origin + (mean * scales) @ directions if len(directions) else origin
        Z = np.where(np.isnan(Z), mode, Z)
    offset = Z - origin
    along = offset @ directions.T
    residual = offset - along @ directions
    q = ((along / scales - mean) / spread) ** 2 if len(directions) else np.zeros((len(Z), 0))
    q = q.sum(axis=1) + (residual ** 2).sum(axis=1) / width ** 2
    return np.exp(-0.5 * q)


def profile(record, steps: int = 61) -> dict:
    """*record*'s shape in its own frame, for its card to draw — evaluated by
    `density`, so the picture is the function the search weighs by.

    One point: the bump's profile through it (`radial`, distance against
    density). Two: the ridge as a field over the position along the line and
    the distance across it (`ridge`). Three: the slab's plane, as a field over
    the two positions in it (`slab`; the width across is the parameter's).
    The points and the peak come along in the same coordinates."""
    origin, directions, scales = frame(record["points"])
    mean, spread, width = params_of(record, len(directions))
    d, along = len(origin), len(directions)

    def other():
        """A direction at right angles to every direction of the frame."""
        for axis in np.eye(d):
            rest = axis - sum((axis @ e) * e for e in directions)
            if np.linalg.norm(rest) > 1e-6:
                return rest / np.linalg.norm(rest)
        return np.zeros(d)

    def reach(j):
        lo, hi = min(-0.25, mean[j] - 3 * spread[j]), max(1.25, mean[j] + 3 * spread[j])
        return np.linspace(lo, hi, steps)

    def rounded(values):
        return [float(f"{v:.4g}") for v in np.asarray(values, float).ravel()]

    if along == 0:
        r = np.linspace(-3 * width, 3 * width, steps)
        Z = origin + np.outer(r, other())
        return {"kind": "radial", "x": rounded(r), "y": rounded(density(Z, record)),
                "width": width}
    if along == 1:
        t, r = reach(0), np.linspace(-3 * width, 3 * width, steps // 2 + 1)
        T, R = np.meshgrid(t, r)
        Z = origin + np.outer(T.ravel() * scales[0], directions[0]) + np.outer(R.ravel(), other())
        field = density(Z, record).reshape(T.shape)
        return {"kind": "ridge", "x": rounded(t), "y": rounded(r),
                "z": [rounded(row) for row in field],
                "points": [[0.0, 0.0], [1.0, 0.0]], "mode": [float(mean[0]), 0.0],
                "width": width}
    t1, t2 = reach(0), reach(1)
    T1, T2 = np.meshgrid(t1, t2)
    Z = origin + np.outer(T1.ravel() * scales[0], directions[0]) \
        + np.outer(T2.ravel() * scales[1], directions[1])
    field = density(Z, record).reshape(T1.shape)
    third = np.asarray(record["points"][2], float) - origin
    return {"kind": "slab", "x": rounded(t1), "y": rounded(t2),
            "z": [rounded(row) for row in field],
            "points": [[0.0, 0.0], [1.0, 0.0],
                       [float(third @ directions[0] / scales[0]), 1.0]],
            "mode": [float(mean[0]), float(mean[1])], "width": width}


def view_coordinates(record, config_space, X) -> np.ndarray:
    """Configurations in ConfigSpace's vectorized form, in the view *record*
    was stated in: their own vectorized values on HP Space's axes, or the frozen
    projection's components."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if record["view"] == "axes":
        order = list(config_space.keys())
        return X[:, [order.index(name) for name in record["hyperparameters"]]]
    encoded = encode_vectors(config_space, _columns(record), X)
    return encoded @ np.asarray(record["map"]["matrix"], dtype=float).T \
        + np.asarray(record["map"]["offset"], dtype=float)


def _columns(record) -> list:
    """The encoded columns of the projection frozen into *record*."""
    return [_Column(label="", name=c["name"], value=c["value"], log=bool(c["log"]),
                    mean=float(c["mean"]), spread=float(c["spread"]))
            for c in record["map"]["columns"]]


def weight(record, config_space, X) -> np.ndarray:
    """*record*'s density at each vectorized configuration in *X*."""
    return density(view_coordinates(record, config_space, X), record)


def check(record, config_space) -> None:
    """Raises `PriorError` for a record that cannot be evaluated against
    *config_space*: an unknown view, axes the space does not have, or points
    that span less than they claim."""
    if record.get("view") not in VIEWS:
        raise PriorError(f"No such view: {record.get('view')!r}.")
    if record["view"] == "axes":
        missing = [h for h in record.get("hyperparameters") or [] if h not in config_space]
        if missing or not record.get("hyperparameters"):
            raise PriorError(f"The search space has no {', '.join(missing) or 'axes'}.")
    elif not record.get("map"):
        raise PriorError("A projected prior needs the projection it was stated in.")
    if len(record["points"][0]) != dimensions(record):
        raise PriorError("The points are not in the view's dimensions.")
    frame(record["points"])


# ── as SMAC takes it ─────────────────────────────────────────────────────────

def _smac_prior_base():
    from smac.acquisition.weight.prior import AbstractInputPrior

    return AbstractInputPrior


def _coarsen(values, resolution):
    from smac.acquisition.weight.prior import _coarsen as coarsen

    return coarsen(values, 1.0, resolution) if resolution is not None else values


def sample_from(config_space, parts, n, rng=None, head=None, also=()) -> list:
    """*n* configurations drawn from the projected priors *parts*, multiplied
    with *head* — a per-hyperparameter prior that draws itself — and with any
    further priors *also*, which are only weighed by.

    Drawn exactly when every part is in HP Space and no two of them, nor
    *head*, say anything about the same hyperparameter: each part's axes are
    drawn from its Gaussian and the rest from *head* or the space. Otherwise
    weighed (`_weighed`)."""
    rng = rng if rng is not None else np.random.RandomState()
    if n <= 0:
        return []
    space = copy.deepcopy(config_space)
    space.seed(int(rng.randint(0, 2 ** 31 - 1)))

    def aimed(count):
        X = _bases(space, head, count, rng)
        for part in parts:
            X = _aim(part.record, config_space, X, rng)
        return X

    out = []
    if not also and _draws_exactly(parts, head):
        for _round in range(_ROUNDS):
            out += _configurations(config_space, aimed(n - len(out)))[0]
            if len(out) >= n:
                return out[:n]
    return out + _weighed(config_space, space, parts, head, also, n - len(out), rng, aimed)


def _draws_exactly(parts, head) -> bool:
    """Whether *parts* can be drawn from directly: all in HP Space, and none
    of them, nor *head*'s tables, about the same hyperparameter twice."""
    seen = set(getattr(head, "tables", None) or {}) if head is not None else set()
    if head is not None and getattr(head, "tables", None) is None:
        return False
    for part in parts:
        record = part.record
        if record["view"] != "axes" or seen & set(record["hyperparameters"]):
            return False
        seen |= set(record["hyperparameters"])
    return True


def _bases(space, head, count, rng) -> np.ndarray:
    """*count* configurations to draw the rest of each draw from, as arrays:
    from *head* when it draws, else from the whole space."""
    drawn = list(head.sample(count, rng) or []) if head is not None else []
    if not drawn:
        drawn = ([space.sample_configuration()] if count == 1
                 else list(space.sample_configuration(count)) if count else [])
    return np.array([c.get_array() for c in drawn], dtype=float).reshape(len(drawn), -1)


def draw_view(record, n, rng) -> np.ndarray:
    """*n* places in the view's coordinates, drawn from the belief's Gaussian
    — the one `density` evaluates: its peak along the points, the spread
    along each of their directions, and the width across."""
    origin, directions, scales = frame(record["points"])
    mean, spread, width = params_of(record, len(directions))
    peak = origin + ((mean * scales) @ directions if len(directions) else 0.0)
    noise = rng.standard_normal((n, len(origin))) * width
    if len(directions):
        noise = noise - (noise @ directions.T) @ directions
        noise = noise + (rng.standard_normal((n, len(directions))) * spread * scales) @ directions
    return peak + noise


def _aim(record, config_space, X, rng) -> np.ndarray:
    if record["view"] == "axes":
        return _aim_axes(record, config_space, X, rng)
    return _aim_projection(record, config_space, X, rng)


def _aim_axes(record, config_space, X, rng) -> np.ndarray:
    """*X* with *record*'s axes drawn from its Gaussian, cut to the space —
    the belief itself, since in HP Space its coordinates are the
    hyperparameters' own. A cell a row leaves inactive stays so."""
    order = list(config_space.keys())
    columns = [order.index(name) for name in record["hyperparameters"]]
    out = np.array(X, dtype=float)
    pending = np.arange(len(out))
    for _round in range(_ROUNDS):
        if not len(pending):
            break
        z = draw_view(record, len(pending), rng)
        inside = ((z >= 0.0) & (z <= 1.0)).all(axis=1)
        _place(out, pending[inside], columns, z[inside])
        pending = pending[~inside]
    if len(pending):
        # A belief mostly outside the space: the nearest places inside it.
        _place(out, pending, columns, np.clip(draw_view(record, len(pending), rng), 0.0, 1.0))
    return _snapped(config_space, out, columns)


def _place(out, rows, columns, z) -> None:
    block = out[np.ix_(rows, columns)]
    out[np.ix_(rows, columns)] = np.where(np.isnan(block), np.nan, z)


def _aim_projection(record, config_space, X, rng) -> np.ndarray:
    """*X* moved towards places drawn from *record*'s Gaussian in its PCA or
    PLS view: each row's numeric values moved the shortest way to ones that
    project onto its drawn place, and read back. Its choices stay as they
    were — a choice is all or nothing, and rounding a moved one-hot column
    undoes the aim — and reading back keeps values inside the space, so a row
    can land short of its place; `_weighed` corrects for that, and keeps the
    rows that landed nearest to draw around."""
    matrix = np.asarray(record["map"]["matrix"], dtype=float)
    offset = np.asarray(record["map"]["offset"], dtype=float)
    columns = _columns(record)
    movable = np.array([c.value is None and c.name in config_space
                        and getattr(config_space[c.name], "choices", None) is None
                        and getattr(config_space[c.name], "sequence", None) is None
                        for c in columns])
    X = np.array(X, dtype=float)
    if not movable.any():
        return X
    inverse = np.linalg.pinv(matrix[:, movable]).T
    z = draw_view(record, len(X), rng)
    for _step in range(_LIFTS):
        encoded = encode_vectors(config_space, columns, X)
        encoded[:, movable] += (z - (encoded @ matrix.T + offset)) @ inverse
        X = _decoded(config_space, columns, encoded, X)
    return X


def _decoded(config_space, columns, encoded, X) -> np.ndarray:
    """*X* with what *encoded* says read back into it: a numeric column's value
    unstandardised, unlogged and kept in its range; a categorical's choice the
    one whose column reads highest. Only where a row has the hyperparameter
    active; one the projection does not encode keeps the row's value. (The
    aim moves only numeric columns, so a choice reads back as it was.)"""
    from ConfigSpace.hyperparameters import IntegerHyperparameter

    order = list(config_space.keys())
    out = X.copy()
    groups: Dict[str, List[int]] = {}
    for j, column in enumerate(columns):
        if column.name in config_space:
            groups.setdefault(column.name, []).append(j)
    for name, indices in groups.items():
        hp = config_space[name]
        k = order.index(name)
        rows = np.flatnonzero(np.isfinite(X[:, k]))
        if not len(rows):
            continue
        choices = getattr(hp, "choices", None) or getattr(hp, "sequence", None)
        first = columns[indices[0]]
        if choices is None and first.value is None:
            if not first.spread:
                continue
            raw = encoded[rows, indices[0]] * first.spread + first.mean
            values = np.clip(np.power(10.0, raw) if first.log else raw, hp.lower, hp.upper)
            if isinstance(hp, IntegerHyperparameter):
                values = np.rint(values)
            out[rows, k] = np.asarray(hp.to_vector(values), dtype=float)
        elif choices is not None:
            named = [(j, choice) for j in indices for choice in choices
                     if str(choice) == columns[j].value]
            if not named:
                continue
            best = encoded[np.ix_(rows, [j for j, _ in named])].argmax(axis=1)
            picked = np.array([named[b][1] for b in best], dtype=object)
            out[rows, k] = np.asarray(hp.to_vector(picked), dtype=float)
    return out


def _snapped(config_space, X, columns) -> np.ndarray:
    """*X* with each of *columns*' active cells at a value its hyperparameter
    can take — a whole number for an integer — so the array SMAC weighs is the
    configuration's own."""
    order = list(config_space.keys())
    for k in columns:
        hp = config_space[order[k]]
        active = np.isfinite(X[:, k])
        if active.any():
            X[active, k] = np.asarray(hp.to_vector(hp.to_value(X[active, k])), dtype=float)
    return X


def _configurations(config_space, X):
    """The rows of *X* that are valid configurations — a row can break a
    condition or a forbidden clause — as configurations, with their row
    numbers."""
    from ConfigSpace import Configuration

    out, kept = [], []
    for i, row in enumerate(X):
        try:
            configuration = Configuration(config_space, vector=row)
            configuration.check_valid_configuration()
        except Exception:  # noqa: BLE001 — whatever it breaks, it is not drawn
            continue
        out.append(configuration)
        kept.append(i)
    return out, kept


def _product(X, head, parts, also) -> np.ndarray:
    """Every prior's density at the rows of *X*, multiplied."""
    out = np.ones(len(X))
    for prior in [p for p in [head, *parts, *also] if p is not None]:
        out = out * np.asarray(prior.pdf(X), dtype=float).reshape(-1)
    return out


def _bandwidths(centres, numeric) -> np.ndarray:
    """How far to draw around the aimed configurations on each numeric axis:
    Scott's rule over them, kept between a hundredth and a quarter of it."""
    spread = np.zeros(centres.shape[1])
    for k in np.flatnonzero(numeric):
        values = centres[np.isfinite(centres[:, k]), k]
        spread[k] = float(np.std(values)) if len(values) > 1 else 0.0
    return np.clip(1.06 * spread * len(centres) ** -0.2, 0.01, 0.25)


def _weighed(config_space, space, parts, head, also, n, rng, aimed) -> list:
    """*n* configurations drawn in proportion to the product of every prior,
    by importance resampling: draws around configurations aimed at the belief,
    and a share from the whole space, each weighed by the priors over how
    likely the drawing made it. Correct however the aimed configurations fall,
    and it reaches a narrow belief because they fall near it.

    Draws are made on SMAC's vectorized axes — numeric values on 0 to 1,
    choices as indices — and weighed there, before an integer is rounded, so
    the weight of each draw is what produced it."""
    if n <= 0:
        return []
    from scipy.special import ndtr, ndtri

    order = list(config_space.keys())
    numeric = np.array([getattr(config_space[name], "choices", None) is None
                        and getattr(config_space[name], "sequence", None) is None
                        and hasattr(config_space[name], "lower") for name in order])
    sizes = np.array([float(getattr(config_space[name], "size", 1) or 1) for name in order])

    # Aimed configurations to draw around: more than are kept, and kept by
    # how much the priors make of them. Which are kept changes nothing about
    # the weighing below, which is against the ones kept, only how many of
    # the draws count.
    wanted = max(16, min(_CENTRES, n))
    centres = aimed(4 * wanted)
    centres = centres[_configurations(config_space, centres)[1]] if len(centres) else centres
    if len(centres) > wanted:
        fit = _product(centres, head, parts, also)
        centres = centres[np.argsort(-fit, kind="stable")[:wanted]]
    if not len(centres):
        centres = _bases(space, head, 16, rng)
    bandwidth = _bandwidths(centres, numeric)

    pool = max(_POOL * n, 64)
    n_uniform = int(round(pool * _UNIFORM_SHARE))
    uniform = _bases(space, None, n_uniform, rng)
    uniform[np.isfinite(uniform) & numeric] = rng.random_sample(
        int((np.isfinite(uniform) & numeric).sum()))
    picked = centres[rng.randint(0, len(centres), pool - n_uniform)]
    around = picked.copy()
    for k in np.flatnonzero(numeric):
        rows = np.flatnonzero(np.isfinite(picked[:, k]))
        if not len(rows):
            continue
        h, c = bandwidth[k], picked[rows, k]
        low, high = ndtr((0.0 - c) / h), ndtr((1.0 - c) / h)
        around[rows, k] = np.clip(c + h * ndtri(low + rng.random_sample(len(rows)) * (high - low)),
                                  0.0, 1.0)
    drawn = np.vstack([uniform, around])

    # How likely the drawing made each draw, with the space's own uniform
    # measure — one over the choices of every active categorical — as the
    # common ground the priors are densities against.
    base = np.prod(np.where(np.isfinite(drawn) & ~numeric, 1.0 / sizes, 1.0), axis=1)
    kernel = np.zeros(len(drawn))
    shape = np.where(np.isfinite(drawn) & ~numeric, drawn, -1.0) + np.where(np.isnan(drawn), -2.0, 0.0)
    centre_shape = np.where(np.isfinite(centres) & ~numeric, centres, -1.0) + \
        np.where(np.isnan(centres), -2.0, 0.0)
    for start in range(0, len(drawn), 2048):
        rows = slice(start, start + 2048)
        same = (shape[rows, None, :] == centre_shape[None, :, :]).all(axis=2)
        log_k = np.zeros(same.shape)
        for k in np.flatnonzero(numeric):
            c = centres[:, k]
            on = np.isfinite(c)
            if not on.any():
                continue
            h = bandwidth[k]
            mass = np.where(on, ndtr((1.0 - np.nan_to_num(c)) / h) - ndtr((0.0 - np.nan_to_num(c)) / h), 1.0)
            x = np.nan_to_num(drawn[rows, k])[:, None]
            term = -0.5 * ((x - np.nan_to_num(c)[None, :]) / h) ** 2 - np.log(h * math.sqrt(2 * math.pi) * mass)
            log_k += np.where(on[None, :], term, 0.0)
        kernel[rows] = (same * np.exp(log_k)).sum(axis=1) / len(centres)
    proposal = _UNIFORM_SHARE * base + (1.0 - _UNIFORM_SHARE) * kernel

    X = _snapped(config_space, drawn.copy(), list(np.flatnonzero(numeric)))
    configurations, kept = _configurations(config_space, X)
    if not configurations:
        return []
    X = X[kept]
    target = _product(X, head, parts, also)
    weights = np.nan_to_num(target * base[kept] / np.maximum(proposal[kept], 1e-300))
    if weights.sum() <= 0:
        return [configurations[i] for i in rng.randint(0, len(configurations), n)]
    chosen = rng.choice(len(configurations), size=n, replace=True, p=weights / weights.sum())
    return [configurations[i] for i in chosen]


def smac_prior(config_space, record):
    """*record* as a prior SMAC's weight layer takes: an `AbstractInputPrior`
    over the whole space, so it can say something about combinations."""
    base = _smac_prior_base()

    class ProjectedPrior(base):
        """One projected prior, evaluated on SMAC's vectorized configurations."""

        #: Samples only by resampling, so a combined prior draws from the
        #: others first — see `combined_prior`.
        projected = True

        def __init__(self):
            check(record, config_space)
            self._names = list(config_space.keys())

        @property
        def hyperparameter_names(self) -> List[str]:
            return self._names

        @property
        def meta(self) -> Dict[str, Any]:
            data = super().meta
            data["projected"] = {"view": record["view"], "points": len(record["points"])}
            return data

        def pdf(self, X, resolution: Optional[int] = None):
            values = weight(record, config_space, X)
            return _coarsen(values, resolution).reshape(-1, 1)

        def sample(self, n, rng=None):
            return sample_from(config_space, [self], n, rng)

    prior = ProjectedPrior()
    prior.record = record
    return prior


def combined_prior(config_space, parts):
    """The per-hyperparameter prior and every projected one as one prior: their
    product, so every belief has to hold at once — which is how the page
    weights them too. SMAC's own ensemble would sum separate priors instead.

    Sampled by `sample_from`, with the per-hyperparameter prior, when there is
    one, drawing everything the projected ones say nothing about."""
    base = _smac_prior_base()
    parts = list(parts)

    class CombinedPrior(base):
        """Several priors over the same space, multiplied."""

        @property
        def hyperparameter_names(self) -> List[str]:
            return list(config_space.keys())

        @property
        def meta(self) -> Dict[str, Any]:
            data = super().meta
            data["parts"] = [part.meta for part in parts]
            return data

        def pdf(self, X, resolution: Optional[int] = None):
            values = np.ones((len(np.atleast_2d(X)), 1))
            for part in parts:
                values = values * np.asarray(part.pdf(X, resolution)).reshape(-1, 1)
            return values

        def sample(self, n, rng=None):
            head = next((p for p in parts if not getattr(p, "projected", False)), None)
            projected = [p for p in parts if getattr(p, "projected", False)]
            also = [p for p in parts if p is not head and p not in projected]
            return sample_from(config_space, projected, n, rng, head=head, also=also)

    return CombinedPrior()
