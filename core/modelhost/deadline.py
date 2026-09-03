"""How long one call to a model may be given.

A run has always enforced a single number on every trial of it —
`DEFAULT_TRIAL_TIMEOUT`, generous on purpose, a "this is wedged" line rather
than a budget. That is the right shape while nobody knows the model, and the
wrong one once someone does: the number that catches a hung fit on a small
table is minutes away from the number that catches one on a large table, and a
search over `n_estimators` spans both inside a single run.

So the deadline becomes a policy, consulted per call rather than fixed at the
top of the run. Two of them:

`FixedDeadline` is the behaviour there has always been, named and given a
zero that means "no limit".

`PredictedDeadline` fits a RandomForest over the durations this run has already
measured and allows each configuration `factor` times what the forest predicts
for that configuration.

**Prediction-relative, and deliberately not median-relative.** A cap set as a
multiple of the typical trial adapts by itself and is biased: duration tracks
capacity — `n_estimators=500` is legitimately 50x `n_estimators=10` — and
capacity tracks performance, so a median-relative cap systematically prunes the
high-capacity end of the space, which is often where the best configuration is.
It kills a trial for being big. A prediction conditioned on the configuration
does not: one predicted to take ten minutes gets ten minutes, and one predicted
at ten seconds and still running after five minutes is stuck.

**What it cannot do is extrapolate.** A forest's prediction never leaves the
range of the targets it was trained on, so a configuration much larger than
anything measured yet is predicted at roughly the slowest thing seen so far.
That is what `ceiling` is for — the fixed deadline, still asked for, still the
outer bound — and `floor` is for the opposite end, where a configuration
predicted at a twentieth of a second would otherwise be given a deadline
shorter than the round trip that carries it.

**And it only ever learns from calls that finished.** A call that timed out is
censored — all it says is "longer than the deadline it was given" — and one
that raised died early rather than ran. Training on either would teach the
forest that the region is fast and tighten the deadline over it, which is the
one feedback loop worth refusing outright. The cost of that choice is that a
configuration killed once contributes nothing towards understanding why, which
is why `ceiling` is an absolute bound and not itself predicted.
"""

from __future__ import annotations

#: Generous on purpose. Measured trials run 0.07-4.7s, so this is not a
#: performance budget — it is the line past which a model is presumed wedged
#: rather than slow, and a real dataset may legitimately sit in `fit` for
#: minutes.
DEFAULT_TRIAL_TIMEOUT = 600.0

#: What a deadline of "none" is, arithmetically. `_await` compares against it
#: and subtracts from it, so a sentinel that behaves like a number is worth more
#: than a `None` every caller has to branch on.
NO_LIMIT = float("inf")

#: How many multiples of the predicted duration a call is allowed. Five rather
#: than two: the prediction is a forest's mean over trees, so half the
#: configurations it has actually seen are slower than their own prediction,
#: and the margin has to cover that as well as the genuine variance between two
#: fits of the same configuration.
DEFAULT_FACTOR = 5.0

#: No predicted deadline is ever shorter than this, however fast the forest
#: thinks a configuration is. A round trip to another process costs a JSON
#: encode, a pipe write and a scheduler wakeup at each end, and the fastest
#: measured trials are already inside a tenth of a second — a deadline derived
#: from those without a floor would be enforcing the noise.
MIN_PREDICTED_DEADLINE = 30.0

#: Completed calls needed before the forest is trusted with a kill decision.
#: Two is enough to *fit* one (`fit_surrogate` accepts that) and nowhere near
#: enough to act on it. Five keeps the earliest, least-informed trials on the
#: ceiling — which is exactly the fixed timeout they would have had anyway —
#: while leaving most of a thirty-trial run adaptive.
MIN_OBSERVATIONS = 5

#: Refit once the observations have grown by this much since the last fit,
#: rather than after every call. A forest over a few hundred rows is tens of
#: milliseconds, which is real against a trial that takes 0.07s, and a duration
#: model does not move discontinuously — the fit from 40 observations is a fine
#: predictor at 45. Growth-proportional rather than every-nth so the cost stays
#: logarithmic in the length of the run.
REFIT_GROWTH = 1.25

#: The forest itself, matching `core.optimizers.base.fit_surrogate` — same
#: estimator, same tree count, fitted against duration instead of a score.
DURATION_TREES = 100


class FixedDeadline:
    """The same number for every call.

    What a bare float handed to `model_session` means, kept as a class so the
    two policies answer the same three questions and the client has one shape to
    talk to.
    """

    #: Read by `RemoteModel`, so building a search space this will not look at
    #: is skipped rather than paid for once a run.
    needs_config_space = False

    __slots__ = ("seconds",)

    def __init__(self, seconds: float | None = DEFAULT_TRIAL_TIMEOUT):
        # 0 and None both read as "no limit", matching the run form, where an
        # empty box has never meant "stop immediately".
        self.seconds = NO_LIMIT if not seconds or seconds <= 0 else float(seconds)

    def bind(self, config_space) -> None:
        """Nothing here needs the search space."""

    def seconds_for(self, config) -> float:
        return self.seconds

    def observe(self, config, elapsed: float, *, completed: bool) -> None:
        """Nothing here learns."""

    def __repr__(self) -> str:
        return f"FixedDeadline({self.seconds!r})"


class PredictedDeadline:
    """`factor` times what a forest predicts this configuration will take.

    Bounded below by `floor` and above by `ceiling`, and equal to `ceiling`
    until `MIN_OBSERVATIONS` calls have completed — so a run that never gets
    that far behaves exactly as `FixedDeadline(ceiling)` would have.
    """

    needs_config_space = True

    def __init__(self, *, factor: float = DEFAULT_FACTOR,
                 ceiling: float | None = DEFAULT_TRIAL_TIMEOUT,
                 floor: float = MIN_PREDICTED_DEADLINE,
                 seed: int = 0):
        self.factor = max(1.0, float(factor))
        self.ceiling = NO_LIMIT if not ceiling or ceiling <= 0 else float(ceiling)
        self.floor = max(0.0, float(floor))
        self.seed = seed
        self._space = None
        self._rows: list = []          # one internal-representation vector per completed call
        self._durations: list = []
        self._forest = None
        self._fitted_at = 0            # len(self._rows) when the forest was last fitted

    # ── the three questions ──────────────────────────────────────────────────

    def bind(self, config_space) -> None:
        """The space configurations are encoded against.

        Handed over by `model_session` once the child has described itself,
        because that is the first moment there is one. Unbound — a caller that
        never binds — the policy simply never leaves the ceiling, which is a
        safe way to be uninformed rather than a crash.
        """
        self._space = config_space

    def seconds_for(self, config) -> float:
        row = self._encode(config)
        if row is None:
            return self.ceiling
        forest = self._fitted()
        if forest is None:
            return self.ceiling
        try:
            predicted = float(forest.predict([row])[0])
        except Exception:  # noqa: BLE001 — a prediction is never worth failing a trial over
            return self.ceiling
        if predicted <= 0:
            return self.ceiling
        return min(self.ceiling, max(self.floor, self.factor * predicted))

    def observe(self, config, elapsed: float, *, completed: bool) -> None:
        """Record one call. Only a completed one is a measurement — see module
        docstring on why a timeout and a crash are both refused."""
        if not completed or elapsed <= 0:
            return
        row = self._encode(config)
        if row is None:
            return
        self._rows.append(row)
        self._durations.append(float(elapsed))

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _encode(self, config):
        """A configuration as the vector the forest is fitted over.

        `Configuration.get_array()` — ConfigSpace's own internal representation,
        the same encoding `core.optimizers.base.fit_surrogate` fits the
        analytics surrogate over, so a duration model and a performance model
        see a configuration identically. A configuration the space rejects
        encodes to `None` and takes the ceiling rather than raising: a deadline
        is not the place to discover that a search space has moved.
        """
        if self._space is None:
            return None
        try:
            from ConfigSpace import Configuration

            return Configuration(self._space, values=dict(config)).get_array()
        except Exception:  # noqa: BLE001 — any unrepresentable config falls back
            return None

    def _fitted(self):
        """The forest, refitted if enough has been learned since the last one."""
        n = len(self._rows)
        if n < MIN_OBSERVATIONS:
            return None
        if self._forest is not None and n < self._fitted_at * REFIT_GROWTH:
            return self._forest
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            forest = RandomForestRegressor(n_estimators=DURATION_TREES,
                                           random_state=self.seed)
            forest.fit(np.array(self._rows), np.array(self._durations))
        except Exception:  # noqa: BLE001 — keep whatever fit we already had
            return self._forest
        self._forest, self._fitted_at = forest, n
        return forest

    def __repr__(self) -> str:
        return (f"PredictedDeadline(factor={self.factor!r}, "
                f"ceiling={self.ceiling!r}, observations={len(self._rows)})")


#: How a run says which of the two it wants. Stored on the Run row and posted by
#: the run form; `as_deadline` is the one place it turns into a policy.
MODE_FIXED = "fixed"
MODE_PREDICTED = "predicted"
MODES = (MODE_FIXED, MODE_PREDICTED)


def as_deadline(spec):
    """Whatever a caller passed as `trial_timeout`, as a policy.

    Accepts a policy (returned as-is), a bare number or `None` (the old
    contract, and what every caller that does not care still passes), or the
    `{"mode": ..., "seconds": ..., "factor": ...}` dict a Run stores. Unknown
    modes and unreadable numbers fall back to the fixed deadline rather than
    raising — this is read from a JSON column that predates the field.
    """
    if spec is None or isinstance(spec, (int, float)):
        return FixedDeadline(spec if spec is not None else DEFAULT_TRIAL_TIMEOUT)
    if hasattr(spec, "seconds_for"):
        return spec
    if not isinstance(spec, dict):
        return FixedDeadline()

    seconds = _number(spec.get("seconds"), DEFAULT_TRIAL_TIMEOUT)
    if spec.get("mode") != MODE_PREDICTED:
        return FixedDeadline(seconds)
    return PredictedDeadline(
        factor=_number(spec.get("factor"), DEFAULT_FACTOR),
        ceiling=seconds,
        floor=_number(spec.get("floor"), MIN_PREDICTED_DEADLINE),
        seed=int(_number(spec.get("seed"), 0)),
    )


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
