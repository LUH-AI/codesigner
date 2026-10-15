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

Django-free, like the rest of `core/`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from .projection import _Column, encode_vectors

VIEWS = ("axes", "pca", "pls")

#: The smallest spread or width a prior may have, in its own units. Below it
#: the bump is narrower than anything the search can resolve, and dividing by
#: it is dividing by zero.
_MIN_SCALE = 1e-6

#: How many configurations to draw for every one handed to SMAC's maximizer
#: when sampling from a prior by resampling: enough that a peak covering a
#: twentieth of the space is reached, few enough to stay quick.
_OVERSAMPLE = 20


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
    projection = record["map"]
    columns = [_Column(label="", name=c["name"], value=c["value"], log=bool(c["log"]),
                       mean=float(c["mean"]), spread=float(c["spread"]))
               for c in projection["columns"]]
    encoded = encode_vectors(config_space, columns, X)
    return encoded @ np.asarray(projection["matrix"], dtype=float).T \
        + np.asarray(projection["offset"], dtype=float)


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


def _resample(config_space, target, n, rng, proposal=None) -> list:
    """*n* configurations drawn in proportion to *target* (a function of
    vectorized rows), by drawing many from *proposal* — or uniformly from the
    space — and keeping each with the weight *target* gives it over what the
    proposal already favoured.

    A copy of the space is seeded and sampled, never the space itself: it is
    the search's, and reseeding it would change every draw SMAC makes."""
    rng = rng if rng is not None else np.random.RandomState()
    pool = n * _OVERSAMPLE
    drawn = list(proposal(pool, rng) or []) if proposal is not None else []
    if not drawn:
        space = copy.deepcopy(config_space)
        space.seed(int(rng.randint(0, 2 ** 31 - 1)))
        drawn = space.sample_configuration(pool)
        drawn = drawn if isinstance(drawn, list) else [drawn]
    rows = np.array([c.get_array() for c in drawn], dtype=float)
    weights = np.nan_to_num(np.asarray(target(rows), dtype=float).reshape(-1), nan=0.0)
    total = weights.sum()
    if total <= 0:
        return drawn[:n]
    picked = rng.choice(len(drawn), size=n, replace=True, p=weights / total)
    return [drawn[i] for i in picked]


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
            return _resample(config_space, lambda rows: weight(record, config_space, rows), n, rng)

    return ProjectedPrior()


def combined_prior(config_space, parts):
    """The per-hyperparameter prior and every projected one as one prior: their
    product, so every belief has to hold at once — which is how the page
    weights them too. SMAC's own ensemble would sum separate priors instead.

    Sampled by drawing from the per-hyperparameter prior when there is one,
    since it samples itself exactly, and keeping each draw with the weight the
    projected ones give it."""
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
            weighed = [p for p in parts if p is not head]

            def target(rows):
                out = np.ones(len(rows))
                for part in weighed:
                    out = out * np.asarray(part.pdf(rows)).reshape(-1)
                return out

            proposal = (lambda count, rng_: head.sample(count, rng_)) if head is not None else None
            return _resample(config_space, target, n, rng, proposal=proposal)

    return CombinedPrior()
