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


# ── decay ───────────────────────────────────────────────────────────────────

#: β as a fraction of the trial budget, which is how the literature states it.
#: πBO sets β = N/10 and DynaBO keeps that, ablating β ∈ {N/50, N/25, N/10,
#: N/5, N/2.5} in its Appendix F.9 — ratios 0.02 to 0.4. Stored as the ratio
#: rather than as β itself so it means the same thing across budgets: a prior
#: exported from a 50-trial run and reopened in a 200-trial one keeps its
#: strength *relative to the run*, which is what the parameterisation says.
#:
#: The bounds are the ablated range, opened a little at each end. Narrower
#: would refuse a value the paper itself evaluated; much wider is meaningless,
#: since `PolynomialDecay` needs β > 0 and a ratio near one leaves the
#: exponent above one for most of the run — a prior that never fades.
BETA_RATIO_MIN, BETA_RATIO_MAX = 0.01, 0.5

#: What the field opens on: πBO's ratio, the one DynaBO settled on after the
#: ablation. Its own summary is that "one can make a case for multiple values" —
#: larger β helps an accurate prior and hurts a misleading one, monotonically in
#: both directions — so this is a default and not a recommendation.
BETA_RATIO_DEFAULT = 0.1

#: The ablated grid, named so the figure can mark it and a reader can see which
#: values were actually measured rather than merely allowed.
BETA_RATIOS_ABLATED = (0.02, 0.04, 0.1, 0.2, 0.4)


def beta_for(budget, stated_ratio=None):
    """β itself: the ratio times this run's trial budget.

    Resolved at apply time rather than stored, because the ratio is the part
    that carries meaning between runs. Note the budget here is the *configured*
    one, fixed for a run — not the count of finished trials, which grows and
    once cancelled the step count it divides, drawing a decaying prior getting
    stronger.
    """
    ratio = BETA_RATIO_DEFAULT if stated_ratio is None else float(stated_ratio)
    ratio = min(max(ratio, BETA_RATIO_MIN), BETA_RATIO_MAX)
    return max(budget * ratio, 1e-6)


def decay_record(stated, budget):
    """The schedule and its factor, as `{"shape": ..., "beta_ratio": ...}`.

    Nested rather than flat keys because a Beta prior's shape parameters are α
    and β: a top-level `beta` beside `params: {alpha, beta}` would be two
    different βs in one record, which is a bug waiting on whoever reads it next.

    Accepts a bare shape string, and a `beta` given absolutely, so priors
    stated before this still load — an absolute β is turned back into the ratio
    it represents for this run's budget.
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
    """*stated* — any belief with a `decay` and an `at_trial` — and how strongly
    it still counts after *trials*, for a run of *budget* trials."""
    if not stated:
        return None

    out = dict(stated)
    decay = decay_record(stated.get("decay"), budget)
    shape = decay["shape"]
    beta = beta_for(budget, decay["beta_ratio"])
    out["decay"] = decay
    # The resolved β alongside the ratio, so the figure can show what the
    # schedule is actually using without multiplying the budget itself.
    out["beta"] = beta
    steps = max(trials - int(stated.get("at_trial") or 0), 0)
    try:
        from smac.acquisition.weight import get_decay_schedule

        # β as stated, not as derived. Reading it off the budget on every
        # request made it grow with the run, which cancelled the step count it
        # divides and drew a decaying prior getting stronger.
        out["exponent"] = float(get_decay_schedule(shape, beta).exponent(steps))
    except ImportError:
        # No weight layer, so nothing decays anything — and `_weighting_warning`
        # is already telling the reader the prior cannot bite at all.
        out["exponent"] = 1.0
    out["steps"] = steps
    # Stated to wait, and the wait is not over: the belief has not started
    # counting, so it has not started fading either. Reported rather than
    # quietly computed as 1, because the page says which of the two it is —
    # "not yet counting" and "counting in full" look identical as a number and
    # mean different things about what happens next.
    if stated.get("delay_decay") and inside_design:
        out["exponent"] = 1.0
        out["steps"] = 0
        out["delayed"] = True
    return out


# ── on the figure ───────────────────────────────────────────────────────────

def prior_density(stated, positions, span):
    """A stated prior's density at *positions*, or None if it states nothing.

    The only place a prior's density is computed. The browser used to evaluate
    it as well, in JavaScript, so a drag could redraw at pointer speed — and
    that second implementation was the problem: two Gaussians that had to agree
    forever, where a disagreement would not raise but would weight the search by
    one shape while the reader watched another.

    Nothing needs it in the browser now. Only the freeform prior is editable on
    the panel, and every shape with parameters is set through fields that
    already round-trip, so the density crosses in the same response.

    *positions* are in vectorized space and *span* is the axis they are drawn
    on; `core.priors` states every distribution on the unit interval, so the
    axis is normalized here. That is the one conversion in the system, and it is
    arithmetic rather than a distribution — which is also why a log-scaled
    hyperparameter needs no special case: its vectorized axis is log space
    already, so a Gaussian on it is the log-normal it deserves.
    """
    if not stated or not positions:
        return None
    lo, width = span[0], (span[1] - span[0]) or 1.0
    unit = [(float(x) - lo) / width for x in positions]
    values = density_from(stated.get("kind"), stated.get("params") or {}, unit)
    return [y for _, y in values] if values else None


def prior_densities(stated, meta):
    """The density everywhere this figure needs it: the slice grid and the cloud.

    Both in one payload because both are functions of the same prior, and a
    second request for the second would be a second chance to disagree.
    """
    span = meta.get("span") or [0.0, 1.0]
    cloud = (meta.get("cloud") or {}).get("positions") or []
    return {"grid": prior_density(stated, meta.get("positions") or [], span),
            "cloud": prior_density(stated, cloud, span)}


def initial_design_notice(result):
    """How much of this run was drawn before anything was being optimized.

    The initial design runs before a surrogate exists, so nothing the reader
    states on this figure can reach it — not a prior, not a constraint. In a
    prior probe the first six trials came out identical with and without a
    prior, for exactly this reason. A figure offering a prior while the run is
    still in that phase should say so rather than let it be discovered.

    The size is *read*, never counted from the trials. That refusal has
    regression tests behind it
    (`tests/core/test_initial_points.py::test_the_size_does_not_come_from_counting_what_the_trials_say`)
    for two reasons: the model's default configuration carries an
    initial-design origin while sitting outside `n_configs`, and files written
    before origins existed label every trial with the optimizer's name. So the
    only trustworthy source is the scenario meta the run left behind, lifted
    onto the result by `deserialize_result`.

    None when there is no recorded size — random search, grid search, an
    imported file with no `scenario.json`. Guessing is the thing the test above
    forbids, and a banner is not worth inventing a number for.
    """
    recorded = ((getattr(result, "metadata", None) or {})
                .get("initial_design") or {})
    n_configs = recorded.get("n_configs")
    if n_configs is None:
        return None

    # The design's own count plus whatever was pinned alongside it. SMAC keeps
    # `use_default_config`'s configuration in `additional_configs`, outside
    # `n_configs`, and it is drawn in the same model-free phase.
    size = max(1, int(n_configs)) + len(recorded.get("additional_configs") or [])
    return {"size": size, "trials": len(result.trials),
            "name": recorded.get("name") or "", "done": len(result.trials) >= size}


def inside_initial_design(result) -> bool:
    """Whether a belief stated now would be stated before the search starts.

    True for an experiment that has never been run, and for one still drawing
    its initial design. Both are the same situation for a prior: the design is
    drawn before any surrogate exists, so nothing stated can reach those trials
    — see `initial_design_notice`.

    False when there is no recorded design size but there are trials, which is
    random or grid search or a file from somebody else's output. Those have no
    design phase to be inside of, and guessing one is what
    `initial_design_notice` already refuses to do.
    """
    if result is None or not result.trials:
        return True
    notice = initial_design_notice(result)
    return bool(notice and not notice["done"])
