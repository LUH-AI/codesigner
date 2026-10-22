"""Times as a reader sees them: the export forms by name, a reader's preferred
form, and a time on the page in its experiment's basis
(`experiments/services/timestamps.py`).
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

from django.utils.dateparse import parse_datetime
from django.utils.translation import gettext_lazy
from django.utils.translation import gettext as _

from experiments.models import account
from experiments.services.settings import resolve_settings
from experiments.services.timestamps import (
    ABSOLUTE, DEFAULT, MODES, NONE, RELATIVE, clamp, origin,
)

from .models import ReaderPreferences

#: Each form's name and what it keeps, wherever one is offered.
LABELS = {
    ABSOLUTE: (gettext_lazy("Real Times"), gettext_lazy("UTC times")),
    RELATIVE: (gettext_lazy("Relative Times"),
               gettext_lazy("Times relative to start of first run")),
    NONE: (gettext_lazy("No Times"), gettext_lazy("List events in order only")),
}


def reader_preference(user) -> str:
    """The form *user* exports times in unless an experiment says otherwise."""
    row = ReaderPreferences.objects.filter(user=account(user)).first()
    return row.export_timestamps if row and row.export_timestamps in MODES else DEFAULT


def set_reader_preference(user, mode) -> None:
    if mode in MODES:
        ReaderPreferences.objects.update_or_create(
            user=account(user), defaults={"export_timestamps": mode})



def for_export(exp, user) -> str:
    """The form to preselect when *user* exports *exp*: the experiment's
    setting, else their preference — no more precise than the experiment's
    times are."""
    chosen = resolve_settings(exp).get("export_timestamps") or reader_preference(user)
    return clamp(exp, chosen)


def offset_text(seconds: float) -> str:
    """A duration as a reader says it: "+42 s", "+3 h 12 min", "+2 d 4 h"."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return _("+%(n)d s") % {"n": round(seconds)}
    minutes = int(seconds // 60)
    if minutes < 60:
        return _("+%(n)d min") % {"n": minutes}
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return _("+%(h)d h %(m)d min") % {"h": hours, "m": minutes}
    days, hours = divmod(hours, 24)
    return _("+%(d)d d %(h)d h") % {"d": days, "h": hours}


def shown(exp, when) -> dict | None:
    """*when* as *exp*'s basis allows it to be shown: {"absolute": datetime}
    or {"relative": "+3 h 12 min"}, or None when it is not to be shown or
    there is no time."""
    if when is None:
        return None
    if isinstance(when, str):
        when = parse_datetime(when)
        if when is None:
            return None
    if isinstance(when, (int, float)):
        when = datetime.fromtimestamp(float(when), tz=dt_timezone.utc)
    basis = exp.data.time_basis
    if basis == NONE:
        return None
    if basis == RELATIVE:
        t0 = origin(exp)
        if t0 is None or when < t0:
            return None
        return {"relative": offset_text((when - t0).total_seconds())}
    return {"absolute": when}
