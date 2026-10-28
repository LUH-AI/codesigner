"""`core.priors` computes every prior density from its `kind` and `params`, for
SMAC's weights and the figure alike, so a prior restored from an `.ihpo` applies
as drawn. Densities live on the ConfigSpace-vectorized axis, where a Gaussian on
a log-scaled hyperparameter is a log-normal in native units.

SMAC's `ConfigSpacePrior` could replace the Normal and Beta branches, except
that `UnitScaler.vectorize_size` log-transforms σ on a log axis and collapses
the width. When
`tests/core/test_prior_density.py::test_configspace_still_mangles_sigma_on_a_log_axis`
fails, ConfigSpace has fixed the σ transform, and only the freeform
interpolation needs to stay.
"""
from __future__ import annotations

import math


def density_from(kind: str, params: dict, positions) -> list[list[float]] | None:
    """`density_from` returns `[[position, density], ...]` on the vectorized
    axis, or None for a prior that states nothing. The density stays
    unnormalized, since weights are only compared across configurations and the
    decay exponent applies afterwards.
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
        # The constant Beta function is dropped, and x is clamped because a
        # shape below one is infinite at the ends.
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

    # A constant weight cannot reorder an acquisition function, so `uniform`
    # returns None.
    return None


def _through(points: list[list[float]], xs: list) -> list[float]:
    """`_through` interpolates monotone cubics (Fritsch-Carlson), since a plain
    spline overshoots between points and invents preferences, even negative
    ones.
    """
    n = len(points)
    if n == 1:
        return [points[0][1] for _ in xs]

    h = [points[i + 1][0] - points[i][0] for i in range(n - 1)]
    delta = [(points[i + 1][1] - points[i][1]) / h[i] if h[i] else 0.0
             for i in range(n - 1)]

    # The curve stays within its control points because tangents are zero
    # wherever the data turns.
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


#: β is stored as a fraction of the trial budget, as πBO (β = N/10) and DynaBO's
#: ablation (N/50 to N/2.5) state it, so a prior keeps its strength across
#: budgets. The bounds widen the ablated range of 0.02 to 0.4 only a little,
#: because a ratio near one would keep the exponent above one for most of the
#: run.
BETA_RATIO_MIN, BETA_RATIO_MAX = 0.01, 0.5

#: DynaBO kept πBO's ratio, though larger β helps an accurate prior and hurts a
#: misleading one.
BETA_RATIO_DEFAULT = 0.1

#: The figure marks these measured ratios.
BETA_RATIOS_ABLATED = (0.02, 0.04, 0.1, 0.2, 0.4)


def beta_for(budget, stated_ratio=None):
    """`beta_for` scales the ratio by the configured budget, fixed for a run,
    since a finished-trial count would grow and cancel the decay.
    """
    ratio = BETA_RATIO_DEFAULT if stated_ratio is None else float(stated_ratio)
    ratio = min(max(ratio, BETA_RATIO_MIN), BETA_RATIO_MAX)
    return max(budget * ratio, 1e-6)


def decay_record(stated, budget):
    """`decay_record` nests the factor because a Beta prior already has
    `params.beta`. A bare shape string and an absolute `beta` still load, the
    latter converted to a ratio of the run's budget.
    """
    if isinstance(stated, str) or stated is None:
        shape, ratio = (stated or "none"), None
    else:
        stated = stated or {}
        shape = stated.get("shape") or "none"
        ratio = stated.get("beta_ratio")
        if ratio is None and stated.get("beta") is not None:
            ratio = float(stated["beta"]) / (budget or 1)

    if ratio is None:
        ratio = BETA_RATIO_DEFAULT
    return {"shape": shape,
            "beta_ratio": min(max(float(ratio), BETA_RATIO_MIN), BETA_RATIO_MAX)}


def with_decay(stated, trials, budget, inside_design=False):
    """`with_decay` adds the decay exponent to *stated* after *trials* of
    *budget*."""
    if not stated:
        return None

    out = dict(stated)
    decay = decay_record(stated.get("decay"), budget)
    shape = decay["shape"]
    beta = beta_for(budget, decay["beta_ratio"])
    out["decay"] = decay
    # The resolved β spares the figure a multiplication.
    out["beta"] = beta
    steps = max(trials - int(stated.get("at_trial") or 0), 0)
    try:
        from smac.acquisition.weight import get_decay_schedule

        out["exponent"] = float(get_decay_schedule(shape, beta).exponent(steps))
    except ImportError:
        # Without SMAC's weight layer nothing decays, and `_weighting_warning`
        # already tells the reader.
        out["exponent"] = 1.0
    out["steps"] = steps
    # `with_decay` sets `delayed` for a delayed belief inside the initial
    # design, since an exponent of 1 alone would read as counting in full.
    if stated.get("delay_decay") and inside_design:
        out["exponent"] = 1.0
        out["steps"] = 0
        out["delayed"] = True
    return out


def prior_density(stated, positions, span):
    """`prior_density` maps *positions* from the *span* axis onto the unit
    interval and evaluates the prior there, or returns None for a prior that
    states nothing. The browser receives the result in the response, so
    `core.priors` is the only implementation.
    """
    if not stated or not positions:
        return None
    lo, width = span[0], (span[1] - span[0]) or 1.0
    unit = [(float(x) - lo) / width for x in positions]
    values = density_from(stated.get("kind"), stated.get("params") or {}, unit)
    return [y for _, y in values] if values else None


def prior_densities(stated, meta):
    """`prior_densities` returns grid and cloud in one payload, so the two
    cannot disagree.
    """
    span = meta.get("span") or [0.0, 1.0]
    cloud = (meta.get("cloud") or {}).get("positions") or []
    return {"grid": prior_density(stated, meta.get("positions") or [], span),
            "cloud": prior_density(stated, cloud, span)}


def initial_design_notice(result):
    """SMAC draws the initial design before a surrogate exists, so a prior
    stated during the design cannot reach its trials. The size comes from the
    scenario meta that `deserialize_result` lifts onto the result, because
    counting trials miscounts the default configuration and files written before
    origins existed. Random search, grid search and imported files without
    `scenario.json` get None.
    """
    recorded = ((getattr(result, "metadata", None) or {})
                .get("initial_design") or {})
    n_configs = recorded.get("n_configs")
    if n_configs is None:
        return None

    # SMAC keeps the default configuration in `additional_configs`, outside
    # `n_configs`, yet draws it in the same model-free phase.
    size = max(1, int(n_configs)) + len(recorded.get("additional_configs") or [])
    return {"size": size, "trials": len(result.trials),
            "name": recorded.get("name") or "", "done": len(result.trials) >= size}


def inside_initial_design(result) -> bool:
    """`inside_initial_design` is True for an experiment never run or still in
    its initial design, and False when trials exist without a recorded design
    size.
    """
    if result is None or not result.trials:
        return True
    notice = initial_design_notice(result)
    return bool(notice and not notice["done"])
