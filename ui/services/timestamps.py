"""What an experiment's times are, and how they leave in a file.

A time says when somebody was working, which is theirs to share or not, and
how much of it a file should carry depends on the experiment and on who is
sending it. So a file carries its times in one of three forms:

- **absolute**: wall-clock times in UTC, as they were recorded;
- **relative**: seconds since the first run was launched. How long things took
  and in what order survive; the date and the time of day do not. What came
  before the first run — setting it up, its first priors — has no time at
  all: until something runs, only the order of things says anything;
- **none**: no times at all. The order of things, and how many trials there
  were when each happened, still say what happened in what sequence.

The form is chosen for each export, preselected from the experiment's own
setting, else the reader's preference, else relative. A file says which form
it is in (`snapshot["timestamps"]`), and an experiment opened from one keeps
that form as its `time_basis`: an experiment whose file had only relative
times has nothing more precise to give, so it is shown, and can be exported
again, in that form or a vaguer one.

Times are stored as datetimes whatever the basis. A relative file's are placed
so that its history ends at the moment it was imported, which keeps every
ordering and every duration true and lets later actions on this instance be
recorded as they happen; only the date is invented, and the basis says not to
show it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

ABSOLUTE = "absolute"
RELATIVE = "relative"
NONE = "none"

#: Most precise first.
MODES = (ABSOLUTE, RELATIVE, NONE)

#: Each form's name and what it keeps, wherever one is offered.
LABELS = {
    ABSOLUTE: (gettext_lazy("Real Times"), gettext_lazy("UTC times")),
    RELATIVE: (gettext_lazy("Relative Times"),
               gettext_lazy("Times relative to start of first run")),
    NONE: (gettext_lazy("No Times"), gettext_lazy("List events in order only")),
}

#: What a reader who has said nothing gets: no wall-clock time leaves the
#: machine, and nothing about the experiment's course is lost.
DEFAULT = RELATIVE


def _rank(mode):
    return MODES.index(mode) if mode in MODES else MODES.index(NONE)


def allowed(exp) -> tuple[str, ...]:
    """The forms *exp*'s times can be exported in: its own, or vaguer."""
    return MODES[_rank(exp.data.time_basis):]


def reader_preference(user) -> str:
    """The form *user* exports times in unless an experiment says otherwise."""
    from ..models import ReaderPreferences

    row = ReaderPreferences.objects.filter(user=_reader(user)).first()
    return row.export_timestamps if row and row.export_timestamps in MODES else DEFAULT


def set_reader_preference(user, mode) -> None:
    from ..models import ReaderPreferences

    if mode in MODES:
        ReaderPreferences.objects.update_or_create(
            user=_reader(user), defaults={"export_timestamps": mode})


def _reader(user):
    return user if getattr(user, "is_authenticated", False) else None


def for_export(exp, user) -> str:
    """The form to preselect when *user* exports *exp*: the experiment's
    setting, else their preference — no more precise than the experiment's
    times are."""
    from .settings import resolve_settings

    chosen = resolve_settings(exp).get("export_timestamps") or reader_preference(user)
    return clamp(exp, chosen)


def clamp(exp, mode) -> str:
    """*mode*, or the most precise form *exp* has when it asks for more."""
    options = allowed(exp)
    return mode if mode in options else options[0]


def origin(exp):
    """When the first run was launched: what relative times count from. None
    before there is one."""
    first = (exp.runs.exclude(created_at=None).order_by("created_at").first()
             or exp.runs.exclude(started_at=None).order_by("started_at").first())
    if first is None:
        return None
    return first.created_at or first.started_at


# ── on the way out ───────────────────────────────────────────────────────────

def _out(value, mode, t0):
    """An ISO time from the snapshot, in *mode*."""
    if not value or mode == NONE:
        return None
    if mode == ABSOLUTE:
        return value
    when = parse_datetime(value) if isinstance(value, str) else value
    if when is None or t0 is None or when < t0:
        return None
    return round((when - t0).total_seconds(), 3)


def _out_posix(value, mode, t0):
    """A POSIX time (a trial's start or end), in *mode*."""
    if value is None or mode == NONE:
        return None
    if mode == ABSOLUTE:
        return value
    if t0 is None or float(value) < t0.timestamp():
        return None
    return round(float(value) - t0.timestamp(), 3)


def apply(snapshot: dict, mode: str, t0) -> dict:
    """*snapshot* with every time in it in *mode*, and saying so.

    Every time there is: when the experiment began, each run's queue, start,
    end and cancel, each run event and history entry, and each trial's start
    and end. In *relative* form *t0* is `origin`, and a time before it — or
    any time, when nothing has run — is left out. Changed in place and
    returned.
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


# ── on the way in ────────────────────────────────────────────────────────────

def basis_of(snapshot: dict) -> str:
    """What form a file's times are in. A file from before the choice existed
    carries wall-clock times where it carries any."""
    mode = snapshot.get("timestamps")
    return mode if mode in MODES else ABSOLUTE


def _offsets(snapshot):
    """Every relative time in *snapshot*, in seconds."""
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
    """*snapshot* with relative times turned back into datetimes, placed so
    that the last of them is *now* (the import). Absolute and absent times are
    left as they are. Changed in place and returned."""
    if basis_of(snapshot) != RELATIVE:
        return snapshot
    offsets = _offsets(snapshot)
    t0 = (now or timezone.now()) - timedelta(seconds=max(offsets, default=0.0))

    def back(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (t0 + timedelta(seconds=float(value))).isoformat()
        # Already a time: a snapshot taken of an experiment that was itself
        # opened from a relative file carries its anchored times.
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


# ── on the page ──────────────────────────────────────────────────────────────

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
