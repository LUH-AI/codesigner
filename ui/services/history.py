"""The experiment's history: every change to how it searches, when and by whom.

`record` is called wherever such a change is made, and writes one
`ExperimentEvent` with the whole value before and after. Runs are recorded by
their own rows (`Run`); the timeline puts the two together. Everything here
travels in the `.ihpo` as its `history` section, so an experiment opened
elsewhere can say what was done to it.

What counts is what changes the search, not what changes the page: stating a
prior does, switching a figure off does not.
"""

from __future__ import annotations

import copy

from django.utils import timezone
from django.utils.dateparse import parse_datetime

#: Every kind of entry, and what its payload holds. Kinds nothing records yet
#: are here so the file format and the timeline know them before they exist.
KINDS = {
    "created": "configuration — how the experiment was set up",
    "imported": "source, filename, dataset_attached",
    "processing_changed": "step, old, new — a Data Handling choice",
    "prior_stated": "hyperparameter, old (None), new",
    "prior_edited": "hyperparameter, old, new",
    "prior_withdrawn": "hyperparameter, old, new (None)",
    "prior_reset": "hyperparameter, old, new — back to what the last run searched under",
    "optimizer_params_changed": "old, new",
    "optimizer_changed": "old, new — each {name, params}, before the first trial",
    "environment_rebuilt": "old, new — the model environment's lock digest",
    # Reserved: output constraints, once an experiment can state them.
    "constraint_added": "constraint, new",
    "constraint_changed": "constraint, old, new",
    "constraint_removed": "constraint, old",
    # Reserved: Data Handling's recipe steps (augmentation, cleaning).
    "data_step_added": "step, new",
    "data_step_removed": "step, old",
}

#: Edits that arrive as a stream — every drag of a prior's curve saves — and
#: are worth one entry per change of mind rather than one per keystroke.
_PRIOR_KINDS = ("prior_stated", "prior_edited", "prior_withdrawn")


def _trials(exp) -> int:
    return len(((exp.data.result or {}).get("data")) or [])


def _actor(user):
    return user if getattr(user, "is_authenticated", False) else None


def record(exp, kind, *, user=None, run=None, **payload):
    """Write one entry in *exp*'s history. Returns it, or None for a draft:
    what is done while setting one up is part of how it was set up, which its
    `created` entry records whole."""
    from ..models import ExperimentEvent

    if kind not in KINDS:
        raise ValueError(f"unknown history kind {kind!r}")
    if getattr(exp, "draft", False):
        return None
    return ExperimentEvent.objects.create(
        data=exp.data, at=timezone.now(), at_trial=_trials(exp), kind=kind,
        payload=copy.deepcopy(payload), run=run, actor=_actor(user))


def record_prior(exp, hp, old, new, *, user=None):
    """A prior stated, edited or withdrawn — folded into the last entry when
    it is about the same hyperparameter and nothing has run since.

    The curve saves as it is dragged, so a minute's adjusting would otherwise
    be a hundred entries. What a run searches under is what stood when it
    started; until one starts, the edits are one change of mind, kept with the
    value it started from. Ending where it began leaves no entry at all.
    """
    if getattr(exp, "draft", False):
        return None
    last = exp.data.history.order_by("-at", "-id").first()
    folding = (last is not None and last.kind in _PRIOR_KINDS
               and last.payload.get("hyperparameter") == hp
               and not exp.runs.filter(created_at__gt=last.at).exists())
    if folding:
        old = last.payload.get("old")
    if old == new:
        if folding:
            last.delete()
        return None
    kind = ("prior_stated" if old is None else
            "prior_withdrawn" if new is None else "prior_edited")
    if folding:
        last.kind, last.at, last.at_trial = kind, timezone.now(), _trials(exp)
        last.payload = {"hyperparameter": hp, "old": old, "new": copy.deepcopy(new)}
        last.actor = _actor(user) or last.actor
        last.save(update_fields=["kind", "at", "at_trial", "payload", "actor"])
        return last
    return record(exp, kind, user=user, hyperparameter=hp, old=old, new=new)


def configuration(snapshot: dict) -> dict:
    """What a `created` entry keeps: everything that decides the search, as
    the file states it — not the results, not where the dataset lived."""
    kept = {key: copy.deepcopy(snapshot[key])
            for key in ("seed", "dataset", "model", "evaluation", "metrics", "optimizer",
                        "priors")
            if key in snapshot}
    for section in ("dataset", "model"):
        if isinstance(kept.get(section), dict):
            kept[section].pop("path", None)
    return kept


# ── in the file ──────────────────────────────────────────────────────────────

def serialized(exp) -> list[dict]:
    """The `history` section: each entry, with the run it belongs to by its
    index in the file's `runs`."""
    index = {pk: i for i, pk in enumerate(exp.runs.order_by("id").values_list("pk", flat=True),
                                          start=1)}
    return [{"kind": e.kind,
             "at": e.at.isoformat() if e.at else None,
             "at_trial": e.at_trial,
             "payload": e.payload,
             "run": index.get(e.run_id)}
            for e in exp.data.history.order_by("at", "id")]


def restore(exp, entries) -> None:
    """Rebuild *exp*'s history from a file's `history` section. Who did each
    thing is not restored, as for runs."""
    from ..models import ExperimentEvent

    runs = dict(enumerate(exp.runs.order_by("id"), start=1))
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            continue
        at = entry.get("at")
        ExperimentEvent.objects.create(
            data=exp.data, at=parse_datetime(at) if isinstance(at, str) else None,
            at_trial=int(entry.get("at_trial") or 0), kind=entry["kind"],
            payload=entry.get("payload") or {}, run=runs.get(entry.get("run")))
