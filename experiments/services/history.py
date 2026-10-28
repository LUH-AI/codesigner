"""`record` writes one `ExperimentEvent` per change to how an experiment
searches, and the events travel in the `.ihpo` `history` section. `record`
counts a stated prior as such a change and skips a hidden figure.
"""

from __future__ import annotations

import copy

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from experiments.models import account

#: Kinds map to payloads; some are reserved before anything records them.
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
    # Reserved for output constraints.
    "constraint_added": "constraint, new",
    "constraint_changed": "constraint, old, new",
    "constraint_removed": "constraint, old",
    # Reserved for recipe steps.
    "data_step_added": "step, new",
    "data_step_removed": "step, old",
}

#: Streamed prior edits fold into one entry per change of mind.
_PRIOR_KINDS = ("prior_stated", "prior_edited", "prior_withdrawn")


def _trials(exp) -> int:
    return len(((exp.data.result or {}).get("data")) or [])


def record(exp, kind, *, user=None, run=None, **payload):
    """`record` returns None for a draft, whose `created` entry holds the setup."""
    from experiments.models import ExperimentEvent

    if kind not in KINDS:
        raise ValueError(f"unknown history kind {kind!r}")
    if getattr(exp, "draft", False):
        return None
    return ExperimentEvent.objects.create(
        data=exp.data, at=timezone.now(), at_trial=_trials(exp), kind=kind,
        payload=copy.deepcopy(payload), run=run, actor=account(user))


def record_prior(exp, hp, old, new, *, user=None):
    """`record_prior` folds an edit into the last entry for the same
    hyperparameter when nothing has run since.
    """
    if getattr(exp, "draft", False):
        return None
    last = exp.data.history.order_by("-at", "-id").first()
    folding = (last is not None and last.kind in _PRIOR_KINDS
               and last.payload.get("hyperparameter") == hp
               and not exp.runs.filter(created_at__gt=last.at).exists())
    if folding and last is not None:
        old = last.payload.get("old")
    if old == new:
        if folding and last is not None:
            last.delete()
        return None
    stated, edited, withdrawn = _PRIOR_KINDS
    kind = stated if old is None else withdrawn if new is None else edited
    if folding and last is not None:
        last.kind, last.at, last.at_trial = kind, timezone.now(), _trials(exp)
        last.payload = {"hyperparameter": hp, "old": old, "new": copy.deepcopy(new)}
        last.actor = account(user) or last.actor
        last.save(update_fields=["kind", "at", "at_trial", "payload", "actor"])
        return last
    return record(exp, kind, user=user, hyperparameter=hp, old=old, new=new)


def configuration(snapshot: dict) -> dict:
    """`configuration` keeps the search-deciding sections, without results or
    file paths."""
    kept = {key: copy.deepcopy(snapshot[key])
            for key in ("seed", "dataset", "model", "evaluation", "metrics", "optimizer",
                        "priors")
            if key in snapshot}
    for section in ("dataset", "model"):
        if isinstance(kept.get(section), dict):
            kept[section].pop("path", None)
    return kept


def serialized(exp) -> list[dict]:
    """`serialized` identifies each entry's run by index in `runs`."""
    index = {pk: i for i, pk in enumerate(exp.runs.order_by("id").values_list("pk", flat=True),
                                          start=1)}
    return [{"kind": e.kind,
             "at": e.at.isoformat() if e.at else None,
             "at_trial": e.at_trial,
             "payload": e.payload,
             "run": index.get(e.run_id)}
            for e in exp.data.history.order_by("at", "id")]


def restore(exp, entries) -> None:
    """`restore` rebuilds the events from *entries* without their actors, like
    runs."""
    from experiments.models import ExperimentEvent, Run

    runs: dict[object, Run] = dict(enumerate(exp.runs.order_by("id"), start=1))
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            continue
        at = entry.get("at")
        ExperimentEvent.objects.create(
            data=exp.data, at=parse_datetime(at) if isinstance(at, str) else None,
            at_trial=int(entry.get("at_trial") or 0), kind=entry["kind"],
            payload=entry.get("payload") or {}, run=runs.get(entry.get("run")))
