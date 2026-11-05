"""`record` writes one `ExperimentEvent` per change to how an experiment
searches, and the events travel in the `.ihpo` `history` section. A prior counts
when a run takes it up (`record_run_priors`), never while the figure is edited.
"""

from __future__ import annotations

import copy

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from experiments.models import account

#: Kinds map to payloads; some are reserved before anything records them.
KINDS = {
    "created": "nothing — the set-up is the file's sections before later changes",
    "imported": "source, filename, dataset_attached",
    "processing_changed": "step, old, new — a Data Handling choice",
    "prior_stated": "hyperparameter, old (None), new — at the run that took it up",
    "prior_edited": "hyperparameter, old, new — at the run that took it up",
    "prior_withdrawn": "hyperparameter, old, new (None) — at the first run without it",
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

#: `PRIOR_KINDS` are written by `record_run_priors` alone.
PRIOR_KINDS = ("prior_stated", "prior_edited", "prior_withdrawn")


def _trials(exp) -> int:
    return len(((exp.data.result or {}).get("data")) or [])


def record(exp, kind, *, user=None, run=None, **payload):
    """`record` returns None for a draft, which has no history until it is
    created."""
    from experiments.models import ExperimentEvent

    if kind not in KINDS:
        raise ValueError(f"unknown history kind {kind!r}")
    if getattr(exp, "draft", False):
        return None
    return ExperimentEvent.objects.create(
        data=exp.data, at=timezone.now(), at_trial=_trials(exp), kind=kind,
        payload=copy.deepcopy(payload), run=run, actor=account(user))


def record_run_priors(run, *, at=None, at_trial=None) -> list:
    """`record_run_priors` writes one entry per hyperparameter whose prior *run*
    takes up differently from the run before it, linked to *run*. Edits that no
    run took up leave no entry.
    """
    exp = run.experiment
    before = (exp.runs.filter(pk__lt=run.pk).order_by("-pk")
              .values_list("priors", flat=True).first()) or {}
    now = run.priors or {}
    stated, edited, withdrawn = PRIOR_KINDS
    entries = []
    for hp in sorted(set(before) | set(now)):
        old, new = before.get(hp), now.get(hp)
        if old == new:
            continue
        kind = stated if old is None else withdrawn if new is None else edited
        entry = record(exp, kind, user=run.started_by, run=run,
                       hyperparameter=hp, old=old, new=new)
        if entry is not None:
            # The entry dates from the run, whenever it is written.
            entry.at = at if at is not None else run.created_at
            entry.at_trial = at_trial if at_trial is not None else entry.at_trial
            entry.save(update_fields=["at", "at_trial"])
        entries.append(entry)
    return entries


def starting_configuration(exp, events) -> dict:
    """`starting_configuration` is the set-up as created: today's sections with
    each later optimizer and Data Handling change undone, from *events* in
    order. Priors are left out, since the runs record them."""
    from .snapshot import snapshot_from_experiment

    found = configuration(snapshot_from_experiment(exp))
    found.pop("priors", None)
    undone: set[str] = set()
    for event in events:
        p = event.payload or {}
        if event.kind == "optimizer_changed" and "optimizer" not in undone:
            undone.add("optimizer")
            found["optimizer"] = copy.deepcopy(p.get("old") or found.get("optimizer"))
        elif event.kind == "processing_changed" and p.get("step") not in undone:
            undone.add(p.get("step"))
            processing = (found.get("dataset") or {}).get("processing")
            if isinstance(processing, dict):
                processing[p.get("step")] = p.get("old")
    return found


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
    runs. Prior entries come from the runs' own priors instead, so a file
    written before `record_run_priors` loses its edits that no run took up."""
    from experiments.models import ExperimentEvent, Run

    runs: dict[object, Run] = dict(enumerate(exp.runs.order_by("id"), start=1))
    for entry in entries or []:
        if (not isinstance(entry, dict) or entry.get("kind") not in KINDS
                or entry["kind"] in PRIOR_KINDS):
            continue
        at = entry.get("at")
        ExperimentEvent.objects.create(
            data=exp.data, at=parse_datetime(at) if isinstance(at, str) else None,
            at_trial=int(entry.get("at_trial") or 0), kind=entry["kind"],
            payload={} if entry["kind"] == "created" else entry.get("payload") or {},
            run=runs.get(entry.get("run")))
    for run in runs.values():
        record_run_priors(run, at_trial=run.trial_offset or 0)
