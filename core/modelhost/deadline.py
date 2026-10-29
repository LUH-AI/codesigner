"""`DEFAULT_TRIAL_TIMEOUT` catches a wedged model, but a hung fit on a small
table and on a large table differ by minutes, and one search over `n_estimators`
spans both. `PredictedDeadline` allows each configuration `factor` times a
RandomForest's prediction of its duration, since a median-relative cap would
prune the slow, high-capacity configurations that often score best. The forest
cannot extrapolate past the slowest duration seen, so `ceiling` bounds the
deadline above and `floor` keeps it above the round-trip cost. The forest only
learns from completed calls, because a timeout or a crash would teach it that a
region is fast.
"""

from __future__ import annotations

#: Measured trials run 0.07-4.7s, and `DEFAULT_TRIAL_TIMEOUT` marks a wedged
#: model, since a real dataset may spend minutes in `fit`.
DEFAULT_TRIAL_TIMEOUT = 600.0

#: `_await` compares against and subtracts from `NO_LIMIT`, so infinity spares
#: callers a `None` branch.
NO_LIMIT = float("inf")

#: A forest predicts the mean over trees, so half the seen configurations run
#: slower than predicted, and the margin also covers run-to-run variance.
DEFAULT_FACTOR = 5.0

#: The fastest trials take under 0.1s, and a deadline below the cost of a
#: process round trip would enforce noise.
MIN_PREDICTED_DEADLINE = 30.0

#: Two calls can fit a forest; five keep the earliest trials on the ceiling
#: while most of a thirty-trial run stays adaptive.
MIN_OBSERVATIONS = 5

#: A forest over a few hundred rows takes tens of milliseconds, real against a
#: 0.07s trial, and growth-proportional refits keep the cost logarithmic in run
#: length.
REFIT_GROWTH = 1.25

#: Matches `core.analytics.fit_surrogate`.
DURATION_TREES = 100


class FixedDeadline:
    """A bare float passed to `model_session` becomes a `FixedDeadline`, so the
    client talks to one policy shape.
    """

    #: `RemoteModel` reads `needs_config_space` to skip building an unused
    #: search space.
    needs_config_space = False

    __slots__ = ("seconds",)

    def __init__(self, seconds: float | None = DEFAULT_TRIAL_TIMEOUT):
        # 0 and None mean no limit, as an empty run-form box does.
        self.seconds = NO_LIMIT if not seconds or seconds <= 0 else float(seconds)

    def bind(self, config_space) -> None:
        """`bind` ignores the space."""

    def seconds_for(self, config) -> float:
        return self.seconds

    def observe(self, config, elapsed: float, *, completed: bool) -> None:
        """`observe` learns nothing."""

    def __repr__(self) -> str:
        return f"FixedDeadline({self.seconds!r})"


class PredictedDeadline:
    """`PredictedDeadline` stays between `floor` and `ceiling`, and equals
    `ceiling` until `MIN_OBSERVATIONS` calls complete.
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
        self._rows: list = []
        self._durations: list = []
        self._forest = None
        self._fitted_at = 0


    def bind(self, config_space) -> None:
        """`RemoteModel` binds the space on first use, from the child's
        greeting. An unbound `PredictedDeadline` stays at the ceiling.
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
        """`observe` records only completed calls (see the module docstring)."""
        if not completed or elapsed <= 0:
            return
        row = self._encode(config)
        if row is None:
            return
        self._rows.append(row)
        self._durations.append(float(elapsed))


    def _encode(self, config):
        """`get_array()` gives the same encoding as
        `core.analytics.fit_surrogate`, and the duration and performance models
        see a configuration alike. `_encode` returns None for a configuration
        the space rejects, and the call gets the ceiling.
        """
        if self._space is None:
            return None
        try:
            from ConfigSpace import Configuration

            return Configuration(self._space, values=dict(config)).get_array()
        except Exception:  # noqa: BLE001 — any unrepresentable config falls back
            return None

    def _fitted(self):
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


#: The Run row stores the mode and the run form posts it; `as_deadline` turns it
#: into a policy.
MODE_FIXED = "fixed"
MODE_PREDICTED = "predicted"
MODES = (MODE_FIXED, MODE_PREDICTED)


DeadlineSpec = float | dict | FixedDeadline | PredictedDeadline | None


def as_deadline(spec) -> FixedDeadline | PredictedDeadline:
    """`as_deadline` accepts a policy, a number or None, or the dict a Run
    stores. Unknown modes and unreadable numbers fall back to a fixed deadline,
    since the JSON column predates the field.
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
