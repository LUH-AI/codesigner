"""Rebuilding a stated prior's density from the numbers that define it.

The browser evaluates a prior's density on the grid and sends it along, so that
what SMAC weights by is literally what the reader saw drawn. That works for as
long as a browser has been involved. A prior restored from an `.ihpo` file has
never been drawn — the file records what the prior *is* (`kind` and `params`),
not a few hundred samples of it — so the density has to be reconstructed here or
the prior silently fails to apply.

That makes this a second implementation of each density, which is a real cost:
`ui/static/ui/acquisition.js` has the first, and a disagreement between them
would not raise, it would quietly change what the optimizer searches while the
figure kept drawing the old shape. Two things hold them together. The formulas
are written in the same space the figure uses — the ConfigSpace-vectorized unit
interval — so there is no conversion to get wrong, only arithmetic. And
`tests/core/test_prior_density.py` pins the properties that would break first.

Everything here is stated on the ConfigSpace-vectorized axis, which is where
the figure states it too. That is not a compromise: for a log-scaled
hyperparameter the vector axis *is* log space, so a Gaussian on it is a
log-normal in native units — which is the prior you want when you believe a
learning rate is "around 1e-3". One formula covers linear and log
hyperparameters alike because the transform is already in the axis.

The alternative was `smac.acquisition.weight.ConfigSpacePrior`, which is the
class SMAC provides for exactly this and would need no formulas at all. It
wants the same distribution this module computes —
`NormalFloatHyperparameter(log=True)` is a Gaussian on the log axis, matching
what the figure draws to about 1e-4 — and for a *linear* hyperparameter the
conversion is exact: μ = `to_value(μ_u)`, σ = σ_u × range, verified to zero
difference.

It is not usable for a log-scaled hyperparameter, and the reason is a defect
rather than a design difference. σ is passed through `UnitScaler.vectorize_size`,
which applies the hyperparameter's log transform to it — but σ is a scale, not
a position, so log-transforming it is a category error. Measured on [1e-4, 1]:
the effective width comes out as |log10(σ)| / log-span, so σ = 1 collapses to a
degenerate spike (width 1.1e-05) and σ = 0.5 and σ = 2.0 are indistinguishable.

`tests/core/test_prior_density.py::test_configspace_still_mangles_sigma_on_a_log_axis`
pins that. When it fails, ConfigSpace has fixed this, and the Normal and Beta
branches below should be deleted in favour of `ConfigSpacePrior` — leaving only
the freeform interpolation, which has no closed form to delegate.
"""
from __future__ import annotations

import math

#: Below this a density is treated as zero. `TabulatedPrior` coarsens what it
#: is given, and a floor keeps a zero from becoming an unreachable region —
#: a prior states a preference, not a constraint.
FLOOR = 1e-12


def density_from(kind: str, params: dict, positions) -> list[list[float]] | None:
    """`[[position, density], ...]` for a stated prior, or None if it states nothing.

    *positions* are the grid points in vectorized space, which is the unit
    interval for a continuous hyperparameter and the choice indices for a
    categorical one — the same axis the figure draws on.

    Unnormalized, deliberately. An acquisition weight is only ever compared
    against itself across configurations, so a constant factor divides out; and
    the decay exponent is applied to whatever comes back, which a normalization
    would not survive intact anyway.
    """
    values = _values_for(kind, params or {}, list(positions))
    if values is None:
        return None
    return [[float(x), max(float(v), 0.0)] for x, v in zip(positions, values)]


def _values_for(kind: str, params: dict, xs: list) -> list[float] | None:
    if kind == "normal":
        mu = float(params.get("mu", 0.5))
        sigma = max(float(params.get("sigma", 0.07)), 1e-6)
        return [math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in xs]

    if kind == "beta":
        # The Beta function itself is left out: it is a constant in x, and a
        # weight that is only ranked cannot tell the difference. Clamped away
        # from the ends because a shape below one is infinite there.
        alpha = max(float(params.get("alpha", 2.0)), 1e-6)
        beta = max(float(params.get("beta", 2.0)), 1e-6)
        out = []
        for x in xs:
            z = min(max(float(x), 1e-6), 1 - 1e-6)
            out.append(math.exp((alpha - 1) * math.log(z)
                                + (beta - 1) * math.log1p(-z)))
        return out

    if kind in ("tabulated", "freeform"):
        points = params.get("points") or []
        if len(points) < 2:
            return None
        return _through(sorted(([float(a), float(b)] for a, b in points),
                               key=lambda p: p[0]), xs)

    # `uniform` states nothing, and so does anything unrecognized: a constant
    # weight cannot reorder an acquisition function, so there is no prior to
    # apply and saying so is better than applying a flat one.
    return None


def _through(points: list[list[float]], xs: list) -> list[float]:
    """Monotone cubic (Fritsch–Carlson) interpolation through *points*.

    The same curve `through()` draws in acquisition.js, and monotone for the
    same reason: a plain cubic spline overshoots between control points, and an
    overshoot in a density is a preference the reader never placed — including,
    below zero, a negative one.
    """
    n = len(points)
    if n == 1:
        return [points[0][1] for _ in xs]

    h = [points[i + 1][0] - points[i][0] for i in range(n - 1)]
    delta = [(points[i + 1][1] - points[i][1]) / h[i] if h[i] else 0.0
             for i in range(n - 1)]

    # Tangents: one-sided at the ends, and zero wherever the data turns, which
    # is what stops the curve running past a control point.
    m = [0.0] * n
    m[0], m[n - 1] = delta[0], delta[n - 2]
    for i in range(1, n - 1):
        m[i] = 0.0 if delta[i - 1] * delta[i] <= 0 else (delta[i - 1] + delta[i]) / 2.0
    for i in range(n - 1):
        if delta[i] == 0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9:
                t = 3.0 / math.sqrt(s)
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]

    out = []
    for x in xs:
        if x <= points[0][0]:
            out.append(points[0][1]); continue
        if x >= points[n - 1][0]:
            out.append(points[n - 1][1]); continue
        i = 0
        while i < n - 2 and x > points[i + 1][0]:
            i += 1
        t = (x - points[i][0]) / h[i] if h[i] else 0.0
        t2, t3 = t * t, t * t * t
        out.append((2 * t3 - 3 * t2 + 1) * points[i][1]
                   + (t3 - 2 * t2 + t) * h[i] * m[i]
                   + (-2 * t3 + 3 * t2) * points[i + 1][1]
                   + (t3 - t2) * h[i] * m[i + 1])
    return out
