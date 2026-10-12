"""The Run form's fields read into what a run is started with: the stopping
criteria, and how long one trial may take.

Plain mappings in, so the page and anything else that fills the form read it
the same way.
"""

from django.conf import settings

from core.metrics import metric_for
from core.modelhost import deadline


# How each stopping criterion is read off the Run form: its parser and the range
# it is held to. Every criterion is here, including the trial cap — the form has
# no required field, only a requirement that at least one be filled in.
STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # None, None: a target is compared against a score, so its range is the
    # metric's and not a constant here — 0 to 1 is accuracy's range, not every
    # metric's. Resolved per submission; see `stopping_from`.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


def text(value) -> str:
    """A field's value as the text a browser would have posted: stripped, and
    "" for nothing. A number arrives as one, so 0 is "0" and not blank."""
    return "" if value is None else str(value).strip()


def stopping_from(data, metric_name: str = "") -> dict:
    """The stopping criteria in *data*, the Run form's fields, ignoring the
    blanks.

    A criterion left empty is absent rather than zero: zero would mean "stop
    immediately", which is never what an empty box asks for. Unparseable input
    is dropped the same way — and if that leaves nothing at all, the caller
    refuses the run rather than starting one that cannot end.

    *metric_name* is the metric the run will optimize, which is what bounds
    `target_score`: a target is compared against a score, so what counts as out
    of range is whatever that metric's range is, and an unbounded metric bounds
    it not at all.
    """
    stopping = {}
    for key, (parse, low, high) in STOPPING_FIELDS.items():
        raw = text(data.get(key))
        if not raw:
            continue
        try:
            value = parse(raw)
        except ValueError:
            continue
        if key == "target_score":
            low, high = metric_for(metric_name).bounds
            # Clamped at both ends rather than dropped below the low one.
            # Dropping made an out-of-range target *absent*, so a run asking for
            # nothing else was then refused for having no criterion at all —
            # which names a different problem from the one the reader created.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def trial_timeout_from(data) -> dict:
    """How long one call to the model may take, from *data*.

    Read separately from `stopping_from` because it is not a stopping
    criterion: nothing here ends a run, it ends a trial, and the run carries on
    with that trial recorded as failed. Storing it among the criteria would also
    mean the collector silently dropped it, since it filters to
    `STOPPING_CRITERIA`.

    Unlike a criterion, this one always has a value — a blank or unreadable box
    falls back to the deployment's `MODEL_TRIAL_TIMEOUT` rather than being
    absent, because "no deadline at all" is a thing to ask for deliberately (0)
    and not a thing to arrive at by mistyping.
    """
    mode = data.get("trial_timeout_mode")
    if mode not in deadline.MODES:
        mode = deadline.MODE_FIXED
    return {
        "mode": mode,
        "seconds": _number(data, "trial_timeout_seconds",
                           settings.MODEL_TRIAL_TIMEOUT, low=0.0),
        "factor": _number(data, "trial_timeout_factor",
                          deadline.DEFAULT_FACTOR, low=1.0),
    }


def _number(data, key: str, default: float, *, low: float) -> float:
    raw = text(data.get(key))
    try:
        return max(low, float(raw.replace(",", ".")))
    except ValueError:
        return float(default)
