"""`timestamps` exports an experiment's times in one of three forms, since
working hours are personal. *absolute* keeps UTC wall-clock times, *relative*
gives seconds since the first run launched, and *none* keeps only the order of
events.

A file states its form in `snapshot["timestamps"]`; an experiment opened from
the file keeps that form as `time_basis` and exports it or a vaguer one. On
import, `anchor` places relative times so the history ends at the import moment,
inventing only the date.
"""

from __future__ import annotations

from datetime import timedelta

from django.utils import timezone
from django.utils.dateparse import parse_datetime

ABSOLUTE = "absolute"
RELATIVE = "relative"
NONE = "none"

#: Precision order.
MODES = (ABSOLUTE, RELATIVE, NONE)

#: Relative keeps the course and hides wall-clock time.
DEFAULT = RELATIVE


def _rank(mode):
    return MODES.index(mode) if mode in MODES else MODES.index(NONE)


def allowed(exp) -> tuple[str, ...]:
    """`allowed` offers the experiment's form or a vaguer one."""
    return MODES[_rank(exp.data.time_basis):]


def clamp(exp, mode) -> str:
    """`clamp` caps *mode* at the most precise form allowed."""
    options = allowed(exp)
    return mode if mode in options else options[0]


def origin(exp):
    """`origin` returns the first run's start, or None before any run."""
    first = (exp.runs.exclude(created_at=None).order_by("created_at").first()
             or exp.runs.exclude(started_at=None).order_by("started_at").first())
    if first is None:
        return None
    return first.created_at or first.started_at


def _out(value, mode, t0):
    if not value or mode == NONE:
        return None
    if mode == ABSOLUTE:
        return value
    when = parse_datetime(value) if isinstance(value, str) else value
    if when is None or t0 is None or when < t0:
        return None
    return round((when - t0).total_seconds(), 3)


def _out_posix(value, mode, t0):
    """`_out_posix` converts a trial's start or end."""
    if value is None or mode == NONE:
        return None
    if mode == ABSOLUTE:
        return value
    if t0 is None or float(value) < t0.timestamp():
        return None
    return round(float(value) - t0.timestamp(), 3)


def apply(snapshot: dict, mode: str, t0) -> dict:
    """`apply` converts every time in *snapshot* to *mode* in place and records
    the mode. In relative form, *t0* is `origin`, and a time before it is
    dropped.
    """
    snapshot["timestamps"] = mode
    snapshot["began_at"] = _out(snapshot.get("began_at"), mode, t0)
    for run in snapshot.get("runs") or []:
        for key in ("created_at", "started_at", "finished_at", "cancel_requested_at"):
            if key in run:
                run[key] = _out(run[key], mode, t0)
        for event in run.get("events") or []:
            if "at" in event:
                event["at"] = _out(event["at"], mode, t0)
    for entry in snapshot.get("history") or []:
        entry["at"] = _out(entry.get("at"), mode, t0)
    for trial in ((snapshot.get("result") or {}).get("data") or []):
        for key in ("starttime", "endtime"):
            if key in trial:
                value = _out_posix(trial[key], mode, t0)
                if value is None:
                    trial.pop(key)
                else:
                    trial[key] = value
    return snapshot


def basis_of(snapshot: dict) -> str:
    """Files from before the choice carry absolute times."""
    mode = snapshot.get("timestamps")
    return mode if mode in MODES else ABSOLUTE


def _offsets(snapshot):
    found = [snapshot.get("began_at")]
    for run in snapshot.get("runs") or []:
        found += [run.get(k) for k in ("created_at", "started_at", "finished_at",
                                       "cancel_requested_at")]
        found += [e.get("at") for e in run.get("events") or []]
    found += [e.get("at") for e in snapshot.get("history") or []]
    for trial in ((snapshot.get("result") or {}).get("data") or []):
        found += [trial.get("starttime"), trial.get("endtime")]
    return [float(v) for v in found if isinstance(v, (int, float)) and not isinstance(v, bool)]


def anchor(snapshot: dict, now=None) -> dict:
    """`anchor` turns relative times into datetimes ending at *now*, the import
    moment, in place."""
    if basis_of(snapshot) != RELATIVE:
        return snapshot
    offsets = _offsets(snapshot)
    t0 = (now or timezone.now()) - timedelta(seconds=max(offsets, default=0.0))

    def back(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (t0 + timedelta(seconds=float(value))).isoformat()
        # Snapshots of an experiment opened from a relative file carry anchored
        # times already.
        return value if isinstance(value, str) else None

    snapshot["began_at"] = back(snapshot.get("began_at")) or t0.isoformat()
    for run in snapshot.get("runs") or []:
        for key in ("created_at", "started_at", "finished_at", "cancel_requested_at"):
            if key in run:
                run[key] = back(run[key])
        for event in run.get("events") or []:
            if "at" in event:
                event["at"] = back(event["at"])
    for entry in snapshot.get("history") or []:
        entry["at"] = back(entry.get("at"))
    for trial in ((snapshot.get("result") or {}).get("data") or []):
        for key in ("starttime", "endtime"):
            if isinstance(trial.get(key), (int, float)):
                trial[key] = t0.timestamp() + float(trial[key])
    return snapshot
