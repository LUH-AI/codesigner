"""`timeline` merges runs and `ExperimentEvent` history into one list, newest
first.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.utils.dateparse import parse_datetime
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from experiments.services.history import PRIOR_KINDS

from . import times
from .figures.data_handling import describe, is_default
from .formatting import sigfigs
from .optimizer_labels import STOPPED_BY_LABELS
from .optimizer_labels import summary as optimizer_summary

_STEPS = {"missing": gettext_lazy("Missing values"), "scale": gettext_lazy("Scaling"),
          "labels": gettext_lazy("Label columns")}
#: A German noun keeps its capital mid-sentence.
_STEPS_IN_SENTENCE = {"missing": gettext_lazy("missing values"), "scale": gettext_lazy("scaling"),
                      "labels": gettext_lazy("label columns")}


def _value(value) -> str:
    if isinstance(value, float):
        return sigfigs(value)
    if value is None:
        return "—"
    return str(value)


_PRIOR_KINDS_SHOWN = {"uniform": gettext_lazy("Uniform"), "normal": gettext_lazy("Normal"), "beta": gettext_lazy("Beta"),
                      "tabulated": gettext_lazy("Freeform")}
_PRIOR_PARAMS = {"mu": "μ", "sigma": "σ", "alpha": "α", "beta": "β"}
_DECAYS = {"logarithmic": gettext_lazy("Logarithmic"), "linear": gettext_lazy("Linear"),
           "quadratic": gettext_lazy("Quadratic"), "cubic": gettext_lazy("Cubic")}


def _decay_shape(prior):
    decay = prior.get("decay")
    shape = (decay.get("shape") or decay.get("kind")) if isinstance(decay, dict) else decay
    return shape if shape and shape != "none" else None


def _prior_params(prior):
    return [(_PRIOR_PARAMS.get(k, k), _value(v))
            for k, v in (prior.get("params") or {}).items() if isinstance(v, (int, float))]


def describe_prior(prior) -> str:
    """`describe_prior` gives "normal (μ 0.5, σ 0.1)"."""
    if not prior:
        return _("none")
    params = ", ".join(f"{name} {value}" for name, value in _prior_params(prior))
    text = prior.get("kind") or "uniform"
    if params:
        text += f" ({params})"
    shape = _decay_shape(prior)
    if shape:
        text += _(", fading (%(kind)s)") % {"kind": shape}
    return text


def prior_rows(prior) -> list:
    if not prior:
        return [(_("Distribution"), _("none"))]
    kind = prior.get("kind") or "uniform"
    rows = [(_("Distribution"), str(_PRIOR_KINDS_SHOWN.get(kind, kind)))]
    rows += _prior_params(prior)
    shape = _decay_shape(prior)
    if shape:
        rows.append((_("Decay"), str(_DECAYS.get(shape, shape))))
        ratio = (prior.get("decay") or {}).get("beta_ratio") if isinstance(prior.get("decay"), dict) else None
        if ratio is not None:
            rows.append(("β / N", _value(ratio)))
        if prior.get("delay_decay"):
            rows.append((_("Delayed until after the initial design"), _("Yes")))
    return rows


#: Run-form order.
_CRITERIA = {
    "max_trials": gettext_lazy("Trials"),
    "target_score": gettext_lazy("Target score"),
    "max_seconds": gettext_lazy("Time limit"),
    "max_trial_seconds": gettext_lazy("Time spent in trials"),
    "no_improvement_trials": gettext_lazy("Trials without improvement"),
    "incumbent_confidence": gettext_lazy("Confidence nothing better remains"),
    "max_failures": gettext_lazy("Failed trials allowed"),
    "max_consecutive_failures": gettext_lazy("Failed in a row allowed"),
}
_SECONDS = {"max_seconds", "max_trial_seconds"}


def _duration(seconds) -> str:
    return times.offset_text(float(seconds)).lstrip("+")


_STOPPED_BY = {**_CRITERIA, "cancelled": gettext_lazy("Cancelled"), "all_failing": gettext_lazy("Too many failures"),
               "exhausted": gettext_lazy("Every configuration tried")}
_STATUSES = {"pending": gettext_lazy("Waiting"), "submitted": gettext_lazy("Submitted"), "running": gettext_lazy("Running"), "done": gettext_lazy("Done"),
             "error": gettext_lazy("Failed"), "cancelled": gettext_lazy("Cancelled")}
_SURROGATES = {"none": gettext_lazy("Kept"), "rebuilt_and_replayed": gettext_lazy("Refitted on the trials so far")}


def _time(exp, when) -> str | None:
    """`_time` formats *when* per *exp*'s basis, or None."""
    from django.utils.dateformat import format as date_format

    shown = times.shown(exp, when)
    if not shown:
        return None
    if "relative" in shown:
        return shown["relative"]
    return date_format(shown["absolute"], "j M Y, H:i:s") + " UTC"


def _yes_no(value) -> str:
    return _("Yes") if value else _("No")


def stopping_rows(stopping) -> list:
    stopping = stopping or {}
    known = [k for k in _CRITERIA if stopping.get(k) is not None]
    other = [k for k in stopping if k not in _CRITERIA and stopping[k] is not None]
    return [(str(_CRITERIA.get(k, k)),
             _duration(stopping[k]) if k in _SECONDS else _value(stopping[k]))
            for k in known + other]


def timeout_rows(timeout) -> list:
    if not timeout:
        return []
    predicted = timeout.get("mode") == "predicted"
    seconds = timeout.get("seconds")
    rows = [(_("Deadline"), _("A multiple of the predicted duration") if predicted
             else _("Fixed"))]
    if seconds is not None:
        rows.append((_("At most") if predicted else _("Limit"),
                     _("no limit") if not seconds else _duration(seconds)))
    if predicted and timeout.get("factor") is not None:
        rows.append((_("Multiple"), "× " + _value(timeout["factor"])))
    return rows


def _entry(exp, kind, at, at_trial, summary, details=(), children=(), actor=None, seq=0,
           scheduled=False):
    """`_entry` drops *details* pairs whose value is empty."""
    shown = []
    for label, value in details:
        if isinstance(value, list):
            if value:
                shown.append({"label": label, "rows": value})
        elif value not in (None, ""):
            shown.append({"label": label, "value": value})
    return {"kind": kind, "at": at, "at_trial": at_trial, "seq": seq,
            "when": times.shown(exp, at), "summary": summary,
            "details": shown, "children": list(children), "actor": actor,
            "scheduled": scheduled}


def _name(user):
    from access.models import shown_name

    return shown_name(user) if user is not None else None


def _configuration_details(configuration):
    model = configuration.get("model") or {}
    dataset = configuration.get("dataset") or {}
    evaluation: Any = configuration.get("evaluation") or {}
    optimizer = configuration.get("optimizer") or {}
    scheme = evaluation.get("scheme")
    split = (_("%(n)s-fold cross-validation") % {"n": evaluation.get("folds")}
             if scheme == "kfold" else
             _("%(n)s backtests") % {"n": evaluation.get("folds")} if scheme == "backtest" else
             _("one split, %(share)s held out") % {"share": _value(evaluation.get("test_size"))})
    return [
        (_("Task"), {"classification": _("Classification"),
                     "regression": _("Regression")}.get(evaluation.get("task"),
                                                        evaluation.get("task"))),
        (_("Model"), model.get("name")),
        (_("Dataset"), dataset.get("demo") or dataset.get("filename")),
        (_("Evaluation"), split),
        (_("Ordered by"), evaluation.get("time_column")),
        (_("Optimizer"), optimizer.get("name")),
        (_("Optimizer settings"), optimizer_summary(optimizer.get("name"), optimizer.get("params"))),
        (_("Seed"), configuration.get("seed")),
    ]


def _data_handling_entry(exp, configuration, at, at_trial, seq):
    """`_data_handling_entry` returns None for configurations recorded before
    `processing` existed."""
    from core.processing import choices_of

    dataset = configuration.get("dataset") or {}
    if "processing" not in dataset:
        return None
    chosen = choices_of(dataset.get("processing"))
    rows = [(str(_STEPS[step]), describe(exp, step, choice, lower=False))
            for step, choice in chosen.items()]
    changed = [_("%(step)s, %(choice)s") % {"step": _STEPS_IN_SENTENCE[step],
                                           "choice": describe(exp, step, choice)}
               for step, choice in chosen.items() if not is_default(exp, step, choice)]
    model = (configuration.get("model") or {}).get("name") or exp.data.model_name
    if not changed:
        summary = _("Data handling: as %(model)s does by default.") % {"model": model}
    elif len(changed) == len(chosen):
        summary = _("Data handling: %(choices)s.") % {"choices": "; ".join(changed)}
    else:
        summary = _("Data handling: %(choices)s; the rest as %(model)s does by default.") % {
            "choices": "; ".join(changed), "model": model}
    return _entry(exp, "data_handling", at, at_trial, summary, rows, seq=seq)


#: `_BELOW_ITS_RUN` exceeds every event id, so run-linked entries sort below
#: the run entry (sequence 0) that shares their time.
_BELOW_ITS_RUN = 10 ** 12


def _event_entry(exp, event, started_as=None):
    p: Any = event.payload or {}
    kind = event.kind
    hp = p.get("hyperparameter")
    if kind == "created":
        configuration = started_as or {}
        summary = _("Set up: %(model)s on %(dataset)s, tuned by %(optimizer)s.") % {
            "model": (configuration.get("model") or {}).get("name") or "?",
            "dataset": ((configuration.get("dataset") or {}).get("demo")
                        or (configuration.get("dataset") or {}).get("filename") or "?"),
            "optimizer": (configuration.get("optimizer") or {}).get("name") or "?"}
        details = _configuration_details(configuration)
    elif kind == "imported":
        summary = (_("Imported from the SMAC run %(name)s.") if p.get("source") == "smac"
                   else _("Imported from %(name)s.")) % {"name": p.get("filename") or "?"}
        if not p.get("dataset_attached"):
            summary += " " + _("Its dataset was not attached.")
        details = [(_("Source"), _("SMAC run") if p.get("source") == "smac" else _(".ihpo file")),
                   (_("File"), p.get("filename")),
                   (_("Dataset attached"), _yes_no(p.get("dataset_attached")))]
    elif kind == "processing_changed":
        summary = _("%(step)s: %(old)s → %(new)s.") % {
            "step": _STEPS.get(p.get("step"), p.get("step")),
            "old": describe(exp, p.get("step"), p.get("old")),
            "new": describe(exp, p.get("step"), p.get("new"))}
        details = [(_("Step"), str(_STEPS.get(p.get("step"), p.get("step")))),
                   (_("Before"), describe(exp, p.get("step"), p.get("old"), lower=False)),
                   (_("After"), describe(exp, p.get("step"), p.get("new"), lower=False))]
    elif kind == "prior_stated":
        summary = (_("Prior to be applied to %(hp)s when the initial design ends: %(new)s.")
                   if p.get("scheduled") else _("Prior stated on %(hp)s: %(new)s.")) % {
            "hp": hp, "new": describe_prior(p.get("new"))}
        details = [(_("Hyperparameter"), hp), (_("Prior"), prior_rows(p.get("new")))]
    elif kind == "prior_edited":
        summary = (_("Prior on %(hp)s to change when the initial design ends.")
                   if p.get("scheduled") else _("Prior on %(hp)s changed.")) % {"hp": hp}
        details = [(_("Hyperparameter"), hp), (_("Before"), prior_rows(p.get("old"))),
                   (_("After"), prior_rows(p.get("new")))]
    elif kind == "prior_withdrawn":
        summary = _("Prior on %(hp)s withdrawn.") % {"hp": hp}
        details = [(_("Hyperparameter"), hp), (_("Was"), prior_rows(p.get("old")))]
    elif kind == "optimizer_params_changed":
        summary = _("Optimizer settings changed.")
        details = [(_("Before"), optimizer_summary(exp.data.optimizer_name, p.get("old"))),
                   (_("After"), optimizer_summary(exp.data.optimizer_name, p.get("new")))]
    elif kind == "optimizer_changed":
        old, new = p.get("old") or {}, p.get("new") or {}
        summary = (_("Optimizer changed: %(old)s → %(new)s.") % {"old": old.get("name"),
                                                                 "new": new.get("name")}
                   if old.get("name") != new.get("name") else _("Optimizer settings changed."))
        details = [(_("Before"), [(str(_("Optimizer")), old.get("name") or "")]
                    + optimizer_summary(old.get("name"), old.get("params"))),
                   (_("After"), [(str(_("Optimizer")), new.get("name") or "")]
                    + optimizer_summary(new.get("name"), new.get("params")))]
    elif kind == "environment_rebuilt":
        summary = _("The model's environment was rebuilt with different packages.")
        details = [(_("Lock before"), (p.get("old") or "")[:12]),
                   (_("Lock after"), (p.get("new") or "")[:12])]
    else:
        summary = kind.replace("_", " ").capitalize() + "."
        details = [(key, _value(value)) for key, value in p.items()]
    # A prior waiting for the initial design has no time.
    if p.get("scheduled"):
        return _entry(exp, kind, None, None, summary, details, actor=_name(event.actor),
                      seq=event.pk or 0, scheduled=True)
    # An entry a run wrote at its start sorts just below that run.
    seq = (event.pk or 0) - (_BELOW_ITS_RUN if event.run_id else 0)
    return _entry(exp, kind, event.at, event.at_trial, summary, details,
                  actor=_name(event.actor), seq=seq)


def _at(value):
    if isinstance(value, str):
        return parse_datetime(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=dt_timezone.utc)
    return value


def _run_event_entry(exp, event):
    kind = event.get("kind")
    at_trial = event.get("at_trial") or 0
    details: list[tuple[str, str | None]] = []
    if kind == "metric_changed":
        summary = _("Optimized metric changed: %(old)s → %(new)s.") % {
            "old": event.get("from"), "new": event.get("to")}
        if event.get("surrogate") == "rebuilt_and_replayed":
            summary += " " + _("The search's model was refitted on the trials so far.")
        details = [(_("From"), event.get("from")), (_("To"), event.get("to")),
                   (_("Search's model"), str(_SURROGATES.get(event.get("surrogate"),
                                                             event.get("surrogate") or "")))]
    elif kind == "prior_applied":
        summary = _("Prior applied to %(hps)s after trial %(n)s.") % {
            "hps": ", ".join(event.get("hyperparameters") or []) or "?", "n": at_trial}
        details = [(_("Hyperparameters"), ", ".join(event.get("hyperparameters") or [])),
                   (_("After trial"), str(at_trial))]
    elif kind == "prior_skipped":
        summary = ((_("Prior refused: %(reason)s.") if event.get("rejected")
                    else _("Prior not applied: %(reason)s.")) % {"reason": event.get("reason")})
        details = [(_("Reason"), event.get("reason")),
                   (_("Refused"), _yes_no(event.get("rejected")))]
    elif kind == "trial_models_not_kept":
        summary = _("%(n)s trials' fitted models were not kept: no room.") % {
            "n": event.get("count")}
        details = [(_("Trials"), str(event.get("count")))]
    else:
        summary = str(kind).replace("_", " ").capitalize() + "."
    details.append((_("At"), _time(exp, _at(event.get("at")))))
    return _entry(exp, kind, _at(event.get("at")), at_trial, summary, details)


def _run_entry(exp, run, number):
    n, offset = run.trial_count, run.trial_offset
    metric = run.primary_metric
    if run.status in ("pending", "submitted"):
        summary = _("Run %(no)s waiting to start, optimizing %(metric)s.")
    elif run.status == "running":
        summary = _("Run %(no)s under way, optimizing %(metric)s.")
    elif not n:
        summary = _("Run %(no)s ran no trials, optimizing %(metric)s.")
    else:
        summary = _("Run %(no)s ran %(n)s trials (%(a)s–%(b)s), optimizing %(metric)s.")
    proposals = [c.get("proposal") for c in run.configurations or [] if c.get("proposal")]
    if proposals and n == 1:
        summary = _("Run %(no)s ran proposal %(proposal)s as trial %(a)s.")
    elif proposals and run.status in run.ACTIVE_STATUSES:
        summary = _("Run %(no)s running proposal %(proposal)s as a trial.")
    summary = summary % {"no": number, "metric": metric, "n": n,
                         "a": (offset or 0) + 1, "b": (offset or 0) + (n or 0),
                         "proposal": ", ".join(str(p) for p in proposals)}
    if run.status == "error":
        summary += " " + _("Then it failed: %(error)s") % {"error": run.error or "?"}
    elif run.status in ("done", "cancelled"):
        reason = STOPPED_BY_LABELS.get(run.stopped_by)
        if reason:
            summary += " " + _("Stopped because %(reason)s.") % {"reason": reason}
    if run.cancel_requested_by is not None:
        summary += " " + _("Cancelled by %(name)s.") % {"name": _name(run.cancel_requested_by)}

    # `._run_record` field order.
    details = [
        (_("Status"), str(_STATUSES.get(run.status, run.status))),
        (_("Queued"), _time(exp, run.created_at)),
        (_("Started"), _time(exp, run.started_at)),
        (_("Finished"), _time(exp, run.finished_at)),
        (_("Cancel requested"), _time(exp, run.cancel_requested_at)),
        (_("Cancelled by"), _name(run.cancel_requested_by)),
        (_("Trial range"), (f"{(offset or 0) + 1}–{(offset or 0) + n}" if n else _("none"))
                      if n is not None else None),
        (_("Optimized metric"), metric),
        (_("Stopping"), stopping_rows(run.stopping)),
        (_("Stopped by"), str(_STOPPED_BY.get(run.stopped_by, run.stopped_by))
                          if run.stopped_by else None),
        (_("Time in trials"), _duration(run.trial_seconds) if run.trial_seconds else None),
        (_("Error"), run.error or None),
        (_("Priors"), [(hp, describe_prior(prior)) for hp, prior in (run.priors or {}).items()]),
        # Configurations picked by hand.
        (_("Proposal"), [(name, _value(value)) for requested in run.configurations or []
                         for name, value in sorted((requested.get("config") or {}).items())]),
        (_("Trial timeout"), timeout_rows(run.trial_timeout)),
        (_("Search settings"), _run_safe_rows(exp, run.optimizer_params)),
    ]
    children = sorted((_run_event_entry(exp, e) for e in run.events or []),
                      key=lambda e: e["at_trial"], reverse=True)
    return _entry(exp, "run", run.created_at or run.started_at,
                  offset if offset is not None else 0, summary, details, children,
                  actor=_name(run.started_by))


def _run_safe_rows(exp, params):
    """`_run_safe_rows` lists the `run_safe` settings a run started with."""
    from core.registry import optimizer_class

    from .optimizer_labels import LABELS

    kind = optimizer_class(exp.data.optimizer_name)
    if not params or kind is None:
        return []
    return [(str(LABELS.get(p.name, (p.label, ""))[0]),
             _value(params.get(p.name)) if params.get(p.name) is not None else _("default"))
            for p in kind.params_schema if p.name in kind.run_safe]


def _scheduled(event, runs):
    """`_scheduled` marks a prior that its run under way only applies once the
    initial design ends."""
    import copy

    new = (event.payload or {}).get("new")
    run = runs.get(event.run_id)
    if (event.kind not in PRIOR_KINDS or not new or not new.get("delay_decay") or run is None
            or run.status not in ("pending", "submitted", "running")
            or any(e.get("kind") == "prior_applied"
                   and event.payload.get("hyperparameter") in (e.get("hyperparameters") or [])
                   for e in run.events or [])):
        return event
    shown = copy.copy(event)
    shown.payload = dict(event.payload, scheduled=True)
    return shown


def build(exp) -> list[dict]:
    from experiments.services import snapshot as adapter
    from experiments.services.history import configuration, starting_configuration

    runs = list(exp.runs.order_by("id").select_related("started_by", "cancel_requested_by"))
    history = list(exp.data.history.select_related("actor").order_by("at", "id"))
    by_pk = {run.pk: run for run in runs}
    started_as = starting_configuration(exp, history)
    entries = [_event_entry(exp, _scheduled(e, by_pk), started_as) for e in history]
    for event in history:
        if event.kind == "created":
            entry = _data_handling_entry(exp, started_as,
                                         event.at, event.at_trial or 0, (event.pk or 0) + 0.5)
            if entry:
                entries.append(entry)
    entries += [_run_entry(exp, run, number) for number, run in enumerate(runs, start=1)]

    if not any(e["kind"] in ("created", "imported") for e in entries):
        snapshot = adapter.snapshot_from_experiment(exp)
        entries.append(_entry(
            exp, "created", exp.data.created_at, 0,
            _("Set up: %(model)s, tuned by %(optimizer)s.") % {
                "model": exp.data.model_name, "optimizer": exp.data.optimizer_name},
            _configuration_details(configuration(snapshot))))
        entry = _data_handling_entry(exp, configuration(snapshot), exp.data.created_at, 0, 0.5)
        if entry:
            entries.append(entry)

    # A file without times sorts by trial count.
    scheduled = [e for e in entries if e["scheduled"]]
    entries = [e for e in entries if not e["scheduled"]]
    timed = [e for e in entries if e["at"] is not None]
    if len(timed) == len(entries):
        entries.sort(key=lambda e: (e["at"], e["seq"]), reverse=True)
    elif not timed:
        entries.sort(key=lambda e: (e["at_trial"], e["seq"]), reverse=True)
    else:
        entries.sort(key=lambda e: ((1, e["at"].timestamp()) if e["at"] else (0, 0.0), e["seq"]),
                     reverse=True)
    # A made-up set-up has no earlier time, so `build` places it last.
    first = ("data_handling", "created")
    return (scheduled + [e for e in entries if e["kind"] not in first]
            + [e for kind in first for e in entries if e["kind"] == kind])
