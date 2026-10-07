"""An experiment's timeline: what was done to it, newest first.

Two records make it up. Runs are their own rows (`Run`), each with the events
that happened inside it — the metric changing at its start, a prior taking
effect at some trial. Everything done between runs is the experiment's history
(`ExperimentEvent`, see ui/services/history.py). This puts the two in one list
and says each entry in a sentence.

An experiment from before the history existed has no `created` entry; one is
made up from what the experiment is, so the list still ends where it began.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

from django.utils.dateparse import parse_datetime
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy

from .formatting import sigfigs
from .optimizer_labels import summary as optimizer_summary
from .services import timestamps

#: Data Handling's steps, by what each is called on the page.
_STEPS = {"missing": gettext_lazy("Missing values"), "scale": gettext_lazy("Scaling"),
          "labels": gettext_lazy("Label columns")}
#: The same, as they read inside a sentence — a translation's own decision, not
#: a lowercasing: a German noun keeps its capital.
_STEPS_IN_SENTENCE = {"missing": gettext_lazy("missing values"), "scale": gettext_lazy("scaling"),
                      "labels": gettext_lazy("label columns")}


def _value(value) -> str:
    if isinstance(value, float):
        return sigfigs(value)
    if value is None:
        return "—"
    return str(value)


#: A prior's distributions and parameters, by the names the prior figure gives
#: them.
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
    """A stated prior in a few words: "normal (μ 0.5, σ 0.1), fading (linear)"."""
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
    """A stated prior as rows, the way the prior figure sets it: distribution,
    its parameters, and how it fades."""
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


#: The stopping criteria, in the run form's order, by short names for what the
#: form asks ("Run until… this many trials have run").
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
    return timestamps.offset_text(float(seconds)).lstrip("+")


#: What ended a run, by the name of what was reached.
_STOPPED_BY = {**_CRITERIA, "cancelled": gettext_lazy("Cancelled"), "all_failing": gettext_lazy("Too many failures"),
               "exhausted": gettext_lazy("Every configuration tried")}
_STATUSES = {"pending": gettext_lazy("Waiting"), "running": gettext_lazy("Running"), "done": gettext_lazy("Done"),
             "error": gettext_lazy("Failed"), "cancelled": gettext_lazy("Cancelled")}
_SURROGATES = {"none": gettext_lazy("Kept"), "rebuilt_and_replayed": gettext_lazy("Refitted on the trials so far")}


def _time(exp, when) -> str | None:
    """*when* as a detail's value, as *exp*'s basis allows: by the clock in
    UTC, as time since the first run, or not at all."""
    from django.utils.dateformat import format as date_format

    shown = timestamps.shown(exp, when)
    if not shown:
        return None
    if "relative" in shown:
        return shown["relative"]
    return date_format(shown["absolute"], "j M Y, H:i:s") + " UTC"


def _yes_no(value) -> str:
    return _("Yes") if value else _("No")


def stopping_rows(stopping) -> list:
    """A run's stopping criteria as rows, each under its name, seconds as a
    duration."""
    stopping = stopping or {}
    known = [k for k in _CRITERIA if stopping.get(k) is not None]
    other = [k for k in stopping if k not in _CRITERIA and stopping[k] is not None]
    return [(str(_CRITERIA.get(k, k)),
             _duration(stopping[k]) if k in _SECONDS else _value(stopping[k]))
            for k in known + other]


def timeout_rows(timeout) -> list:
    """A run's deadline on one model fit as rows: fixed or predicted, the
    outer bound in seconds, and under prediction the multiple allowed."""
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
    """One entry. *details* are `(label, value)` pairs, or `(label, rows)`
    with rows a list of pairs for a value that is itself a short list; empty
    ones are left out. A *scheduled* one has not happened yet, and has no
    time."""
    shown = []
    for label, value in details:
        if isinstance(value, list):
            if value:
                shown.append({"label": label, "rows": value})
        elif value not in (None, ""):
            shown.append({"label": label, "value": value})
    return {"kind": kind, "at": at, "at_trial": at_trial, "seq": seq,
            "when": timestamps.shown(exp, at), "summary": summary,
            "details": shown, "children": list(children), "actor": actor,
            "scheduled": scheduled}


def _name(user):
    return user.get_username() if user is not None else None


# ── what was done between runs ───────────────────────────────────────────────

def _configuration_details(configuration):
    model = configuration.get("model") or {}
    dataset = configuration.get("dataset") or {}
    evaluation = configuration.get("evaluation") or {}
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
    """The Data Handling choices an experiment was set up with, as an entry of
    their own beside its set-up: every step as a row, the summary naming the
    ones that are not the model's own default. None for a configuration from
    before processing was recorded."""
    from core.processing import choices_of
    from datahandling.panels import describe, is_default

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


def _event_entry(exp, event):
    p = event.payload or {}
    kind = event.kind
    hp = p.get("hyperparameter")
    if kind == "created":
        configuration = p.get("configuration") or {}
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
        from datahandling.panels import describe

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
    elif kind == "prior_reset":
        summary = _("Prior on %(hp)s put back to what the last run searched under: "
                    "%(new)s.") % {"hp": hp, "new": describe_prior(p.get("new"))}
        details = [(_("Hyperparameter"), hp), (_("Before"), prior_rows(p.get("old"))),
                   (_("After"), prior_rows(p.get("new")))]
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
    # A prior waiting for the initial design to end has not happened: it is
    # shown as what is coming, with no time.
    if p.get("scheduled"):
        return _entry(exp, kind, None, None, summary, details, actor=_name(event.actor),
                      seq=event.pk or 0, scheduled=True)
    return _entry(exp, kind, event.at, event.at_trial, summary, details,
                  actor=_name(event.actor), seq=event.pk or 0)


# ── the runs ─────────────────────────────────────────────────────────────────

def _at(value):
    if isinstance(value, str):
        return parse_datetime(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=dt_timezone.utc)
    return value


def _run_event_entry(exp, event):
    kind = event.get("kind")
    at_trial = event.get("at_trial") or 0
    details = []
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
    from .views import STOPPED_BY_LABELS

    n, offset = run.trial_count, run.trial_offset
    metric = run.primary_metric
    if run.status == "pending":
        summary = _("Run %(no)s waiting to start, optimizing %(metric)s.")
    elif run.status == "running":
        summary = _("Run %(no)s under way, optimizing %(metric)s.")
    elif not n:
        summary = _("Run %(no)s ran no trials, optimizing %(metric)s.")
    else:
        summary = _("Run %(no)s ran %(n)s trials (%(a)s–%(b)s), optimizing %(metric)s.")
    summary = summary % {"no": number, "metric": metric, "n": n,
                         "a": (offset or 0) + 1, "b": (offset or 0) + (n or 0)}
    if run.status == "error":
        summary += " " + _("Then it failed: %(error)s") % {"error": run.error or "?"}
    elif run.status in ("done", "cancelled"):
        reason = STOPPED_BY_LABELS.get(run.stopped_by)
        if reason:
            summary += " " + _("Stopped because %(reason)s.") % {"reason": reason}
    if run.cancel_requested_by is not None:
        summary += " " + _("Cancelled by %(name)s.") % {"name": _name(run.cancel_requested_by)}

    # What the file's record of the run holds (`snapshot._run_record`), in
    # its order: what the summary says in a sentence, laid out.
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
        (_("Trial timeout"), timeout_rows(run.trial_timeout)),
    ]
    children = sorted((_run_event_entry(exp, e) for e in run.events or []),
                      key=lambda e: e["at_trial"], reverse=True)
    return _entry(exp, "run", run.created_at or run.started_at,
                  offset if offset is not None else 0, summary, details, children,
                  actor=_name(run.started_by))


# ── all of it ────────────────────────────────────────────────────────────────

_PRIOR_KINDS = ("prior_stated", "prior_edited", "prior_withdrawn", "prior_reset")


def _effective(events, runs):
    """The prior entries that influenced the search, rewritten to say how.

    A prior is a statement about where the next trials should go, and only a
    run acting on it makes it part of the experiment. So, per hyperparameter,
    a statement counts once a run applied it (its `prior_applied` event names
    the hyperparameter), or while it is the one standing — including one
    waiting for the initial design to end, which is shown as scheduled. One
    replaced or withdrawn before any run applied it is not shown, and nor is
    a withdrawal that only cancels such a one: it never touched the search.

    What is shown is each change between statements that counted: stated,
    changed or withdrawn, from the last one that counted.
    """
    import copy

    def at_of(run):
        return run.created_at or run.started_at

    def applied(hp, start, end):
        for run in runs:
            when = at_of(run)
            if when is None or start is None or when < start or (end and when >= end):
                continue
            if any(e.get("kind") == "prior_applied" and hp in (e.get("hyperparameters") or [])
                   for e in run.events or []):
                return True
        return False

    def ran_between(start, end):
        return any(at_of(r) and start and at_of(r) >= start and (not end or at_of(r) < end)
                   for r in runs)

    chains = {}
    for event in events:
        chains.setdefault(event.payload.get("hyperparameter"), []).append(event)
    shown = []
    for hp, chain in chains.items():
        standing = None
        for i, event in enumerate(chain):
            new = event.payload.get("new")
            end = chain[i + 1].at if i + 1 < len(chain) else None
            last = i + 1 == len(chain)
            # Without times nothing says which runs came after it, so it is
            # shown rather than hidden on a guess.
            counted = (event.at is None
                       or (applied(hp, event.at, end) if new is not None
                           else ran_between(event.at, end) or last))
            if not (counted or last) or new == standing:
                continue
            shown_event = copy.copy(event)
            payload = dict(event.payload, old=standing)
            if new is not None and last and not applied(hp, event.at, None):
                payload["scheduled"] = bool(new.get("delay_decay"))
            shown_event.payload = payload
            if event.kind != "prior_reset":
                shown_event.kind = ("prior_stated" if standing is None else
                                    "prior_withdrawn" if new is None else "prior_edited")
            shown.append(shown_event)
            standing = new
    return shown


def build(exp) -> list[dict]:
    """Every entry for *exp*, newest first: what is scheduled, then what
    happened."""
    from .services import snapshot as adapter
    from .services.history import configuration

    runs = list(exp.runs.order_by("id").select_related("started_by", "cancel_requested_by"))
    history = list(exp.data.history.select_related("actor").order_by("at", "id"))
    others = [e for e in history if e.kind not in _PRIOR_KINDS]
    priors = _effective([e for e in history if e.kind in _PRIOR_KINDS], runs)
    entries = [_event_entry(exp, e) for e in others + priors]
    # Each set-up's Data Handling, just after it: recorded within it, but a
    # decision about the data rather than about the search's machinery.
    for event in others:
        if event.kind == "created":
            entry = _data_handling_entry(exp, (event.payload or {}).get("configuration") or {},
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

    # By time where there is one, entries made in the same moment in the order
    # they were made (setting an experiment up records several at once). An
    # experiment whose file carried no times is ordered by how many trials
    # there were, which is the same order. One whose file had relative times
    # has none for what came before its first run, so that comes first, in
    # the order it was made.
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
    # The set-up last, its Data Handling just above it, whatever their times
    # say: one made up for an experiment from before the history has none
    # earlier than what came after.
    first = ("data_handling", "created")
    return (scheduled + [e for e in entries if e["kind"] not in first]
            + [e for kind in first for e in entries if e["kind"] == kind])
