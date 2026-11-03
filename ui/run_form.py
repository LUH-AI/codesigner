"""`run_form` parses the Run form for the page and the assistant."""

from django.conf import settings

from core.metrics import metric_for
from core.modelhost import deadline

#: Criterion parsers and ranges.
STOPPING_FIELDS = {
    "max_trials": (int, 1, 100_000),
    # The metric bounds the target.
    "target_score": (float, None, None),
    "max_seconds": (float, 1.0, None),
    "max_trial_seconds": (float, 1.0, None),
    "no_improvement_trials": (int, 1, None),
    "incumbent_confidence": (float, 0.0, 1.0),
    "max_failures": (int, 1, None),
    "max_consecutive_failures": (int, 1, None),
}


UNPLANNED_TRIALS = 30


def opening_trials(budget: int | None, done: int):
    """`opening_trials` gives the run form's opening trial limit."""
    if not budget:
        return UNPLANNED_TRIALS
    return max(budget - done, 0) or ""


def text(value) -> str:
    """`text` mimics a browser post, so 0 stays "0" and None becomes ""."""
    return "" if value is None else str(value).strip()


def stopping_from(data, metric_name: str = "") -> dict:
    """`stopping_from` skips blank and unparseable criteria, since zero would
    stop the run at once.
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
            # Clamping keeps an out-of-range target from vanishing.
            if low is not None:
                value = max(value, low)
        elif value < low:
            continue
        stopping[key] = min(value, high) if high is not None else value
    return stopping


def trial_timeout_from(data) -> dict:
    """`trial_timeout_from` keeps the per-trial deadline out of the stopping
    criteria. A blank box uses `MODEL_TRIAL_TIMEOUT`; 0 turns the deadline off.
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
