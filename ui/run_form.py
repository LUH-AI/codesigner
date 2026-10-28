"""`run_form` reads the Run form's fields from a plain mapping, so the page and
the assistant start a run from the same reading."""

from django.conf import settings

from core.metrics import metric_for
from core.modelhost import deadline

#: Each Run-form criterion's parser and range.
STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # `stopping_from` bounds the target by its metric's range.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


def text(value) -> str:
    """`text` gives a value as a browser would post it, so a number 0 stays
    "0" and a missing value becomes ""."""
    return "" if value is None else str(value).strip()


def stopping_from(data, metric_name: str = "") -> dict:
    """`stopping_from` drops blank and unparseable criteria instead of reading
    them as zero, since zero would stop the run at once. *metric_name* supplies
    the bounds of `target_score`.
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
            # Clamping keeps an out-of-range target from vanishing and tripping
            # the no-criterion refusal.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def trial_timeout_from(data) -> dict:
    """`trial_timeout_from` reads the per-trial deadline apart from the
    criteria, since a timeout fails one trial and the run continues. A blank box
    falls back to `MODEL_TRIAL_TIMEOUT`, and only an explicit 0 turns the
    deadline off.
    """
    mode = data.get("trial_timeout_mode")
    if mode not in deadline.MODES:
        mode = deadline.MODE_FIXED
    return {
        "mode": mode,
        "seconds": number(data, "trial_timeout_seconds", settings.MODEL_TRIAL_TIMEOUT, low=0.0),
        "factor": number(data, "trial_timeout_factor", deadline.DEFAULT_FACTOR, low=1.0),
    }


def number(data, key: str, default: float, *, low: float) -> float:
    raw = text(data.get(key))
    try:
        return max(low, float(raw.replace(",", ".")))
    except ValueError:
        return float(default)
