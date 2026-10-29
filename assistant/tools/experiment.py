"""The tools that answer questions about an experiment from what it stores — its
set-up, runs, priors, history and every trial — rather than from what its pages
draw (assistant/results.py does the analyses).

The experiment is the one named, else the conversation's, else the one whose
page the person has open, reached as the person may view it.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict

from access.policy import VIEW

from ui.permissions import BOTH

from .. import results
from .registry import Refused, ToolResult, ToolSpec

#: Rebuilt trials kept between calls, by what would change them.
_KEPT: OrderedDict = OrderedDict()
_KEPT_SIZE = 8
_kept_lock = threading.Lock()


def experiment_of(context, experiment: int | None = None):
    """The experiment a call is about, as the person may view it."""
    pk = experiment or context.conversation.experiment_id
    if pk is None:
        from .. import acts

        page = acts.current(context.conversation)
        pk = page.experiment_id if page is not None else None
    if pk is None:
        raise Refused("no experiment is named, and neither the conversation nor the person's "
                      "page is about one")
    return context.experiment(pk, VIEW, BOTH)


def _runs(exp) -> list[dict]:
    out = []
    for index, run in enumerate(exp.runs.order_by("pk"), start=1):
        row = {"index": index, "status": run.status, "stopped_by": run.stopped_by or None,
               "stopping": run.stopping or {}, "first": None, "last": None}
        if run.trial_offset is not None and run.trial_count is not None:
            row["first"], row["last"] = run.trial_offset + 1, run.trial_offset + run.trial_count
        out.append(row)
    return out


def trials_of(exp) -> results.Trials:
    """*exp*'s trials, rebuilt from what it stores; kept until a run or the
    stored result changes."""
    from experiments.services.snapshot import config_space_for, rebuild_experiment

    runs = _runs(exp)
    key = (exp.pk, exp.data_id, len(runs), tuple((r["status"], r["last"]) for r in runs),
           json.dumps(exp.data.optimizer_params or {}, sort_keys=True, default=str))
    with _kept_lock:
        if key in _KEPT:
            _KEPT.move_to_end(key)
            return _KEPT[key]
    built = rebuild_experiment(exp)
    if built is None:
        raise Refused("the experiment's results cannot be read on this instance: a metric or "
                      "optimizer it uses is not installed")
    from core.registry import optimizer_class

    kind = optimizer_class(exp.data.optimizer_name)
    found = results.Trials(result=built["result"], space=config_space_for(built),
                           optimizer=kind.name if kind is not None else exp.data.optimizer_name,
                           settings=dict(exp.data.optimizer_params or {}), runs=runs)
    with _kept_lock:
        _KEPT[key] = found
        while len(_KEPT) > _KEPT_SIZE:
            _KEPT.popitem(last=False)
    return found


def _answer(exp, found: dict) -> ToolResult:
    return ToolResult(output=json.dumps(found, default=str, ensure_ascii=False),
                      experiment_id=exp.pk)


def _run(question):
    try:
        return question()
    except results.ResultsError as error:
        raise Refused(str(error))


# ── read_experiment ─────────────────────────────────────────────────────────

def _overview(exp, t) -> dict:
    data = exp.data
    metric = t.result.primary_metric if t.trials else data.current_metric
    out = {"experiment": exp.pk, "name": exp.title, "draft": exp.draft, "task": data.task,
           "dataset": data.dataset_path().name if data.dataset_path() else None,
           "target": data.target_column or None, "model": data.model_name,
           "optimizer": data.optimizer_name, "metrics": list(data.metric_names or []),
           "optimized_metric": metric, "trials": len(t.trials),
           "failed_trials": sum(x.failed for x in t.trials), "runs": len(t.runs),
           "priors_stated_on": sorted(data.priors or {})}
    if metric and t.trials:
        out["higher_is_better"] = t.higher_is_better(metric)
        ranked = t.ranked(metric)
        if ranked:
            out["best_trial"] = t.row(ranked[0], [metric])
    if t.design:
        out["initial_design_size"] = t.design["size"]
    if t.runs:
        out["last_run"] = {k: t.runs[-1][k] for k in ("index", "status", "stopped_by")}
    return out


def _runs_aspect(exp, t) -> dict:
    from ui.optimizer_labels import STOPPED_BY_LABELS

    out = []
    for row, run in zip(t.runs, exp.runs.order_by("pk")):
        out.append({**row, "stopped_because": str(STOPPED_BY_LABELS.get(run.stopped_by, ""))
                    or None, "metric": run.primary_metric,
                    "trial_timeout": run.trial_timeout or None,
                    "priors": sorted(run.priors or {}),
                    "started_at": run.started_at, "finished_at": run.finished_at,
                    "error": (run.error or "")[:500] or None})
    return {"runs": out}


def _settings(exp, t) -> dict:
    from core.registry import optimizer_class

    data = exp.data
    labels = {}
    kind = optimizer_class(data.optimizer_name)
    if kind is not None:
        labels = {p.name: p.label for p in kind.params_schema}
    params = {name: {"value": value, **({"label": labels[name]} if name in labels else {})}
              for name, value in (data.optimizer_params or {}).items() if value is not None}
    space = []
    for hp in t.hyperparameters():
        row = {"name": hp.name, "type": type(hp).__name__,
               "default": getattr(hp, "default_value", None)}
        if hasattr(hp, "lower"):
            row.update({"lower": hp.lower, "upper": hp.upper, "log": bool(hp.log)})
        if hasattr(hp, "choices"):
            row["choices"] = list(hp.choices)
        space.append(row)
    return {"optimizer": data.optimizer_name, "optimizer_settings": params,
            "search_space": space, "seed": data.seed, "cv_folds": data.cv_folds or None,
            "test_size": data.test_size if not data.cv_folds else None,
            "time_column": data.time_column or None, "processing": data.processing or {}}


def _priors(exp, t) -> dict:
    from core.priors import inside_initial_design

    from experiments.services.priors import decayed_prior
    from ui.timeline import describe_prior

    stated = exp.data.priors or {}
    if not stated:
        return {"priors": {}, "note": "No prior is stated: every hyperparameter is searched "
                                      "uniformly."}
    inside = inside_initial_design(t.result if t.trials else None)
    out = {}
    for name, prior in stated.items():
        now = decayed_prior(exp, name, len(t.trials), inside) or {}
        out[name] = {"prior": str(describe_prior(prior)), "stated_at_trial": prior.get("at_trial"),
                     "weight_now": (None if now.get("exponent") is None
                                    else round(float(now["exponent"]), 4)),
                     "trials_since_stated": now.get("steps")}
        if now.get("delayed"):
            out[name]["note"] = "held at full weight until the initial design is done"
    return {"priors": out, "reading": "weight_now is the exponent on the prior: 1 counts in "
                                      "full, 0 not at all."}


def _timeline(exp, t) -> dict:
    from ui.timeline import build

    def plain(entry):
        out = {"what": str(entry["summary"]), "at_trial": entry["at_trial"],
               "when": str(entry["when"]) if entry.get("when") else None}
        details = []
        for detail in entry["details"]:
            if "value" in detail:
                details.append(f"{detail['label']}: {detail['value']}")
            else:
                details.append(f"{detail['label']}: " + "; ".join(
                    " ".join(str(c) for c in (r if isinstance(r, (list, tuple)) else [r]))
                    for r in detail["rows"][:10]))
        if details:
            out["details"] = [str(d)[:300] for d in details[:12]]
        if entry.get("actor"):
            out["by"] = entry["actor"]
        return out

    entries = build(exp)
    return {"entries": [plain(e) for e in entries[:30]], "of": len(entries),
            "order": "newest first"}


ASPECTS = {"overview": _overview, "runs": _runs_aspect, "settings": _settings,
           "priors": _priors, "timeline": _timeline}


def _read_experiment(context, aspect: str = "overview", experiment: int | None = None
                     ) -> ToolResult:
    exp = experiment_of(context, experiment)
    return _answer(exp, ASPECTS[aspect](exp, trials_of(exp)))


_EXPERIMENT = {"type": "integer",
               "description": "Which experiment; by default the conversation's, else the one "
                              "whose page is open."}

READ_EXPERIMENT = ToolSpec(
    name="read_experiment",
    description=(
        "What an experiment is, from what it stores. `aspect`: overview (task, data, model, "
        "optimizer, metric, trial and failure counts, the best trial, the initial design's "
        "size), runs (each run's trials, how it was set to stop and why it stopped), "
        "settings (the optimizer's settings and the search space with bounds), priors (each "
        "stated prior and how much it still counts), timeline (what was done, when, by "
        "whom)."),
    parameters={"type": "object", "properties": {
        "aspect": {"type": "string", "enum": list(ASPECTS)}, "experiment": _EXPERIMENT}},
    handler=_read_experiment,
)


# ── read_trials, compare_trials, analyze_experiment ─────────────────────────

def _read_trials(context, trial: int | None = None, order: str = "best",
                 metric: str = "", limit: int = 10, where: list | None = None,
                 experiment: int | None = None) -> ToolResult:
    exp = experiment_of(context, experiment)
    t = trials_of(exp)
    if trial is not None:
        return _answer(exp, _run(lambda: results.trial(t, int(trial), metric or None)))
    return _answer(exp, _run(lambda: results.rows(t, order, metric or None, limit, where)))


READ_TRIALS = ToolSpec(
    name="read_trials",
    description=(
        "The experiment's trials as stored. With `trial` (its number), everything about that "
        "one: configuration, every score, status and failure, duration, how it arose "
        "(initial design, model-guided, random, requested), its run, its rank and whether it "
        "was a new best when it ran. Otherwise a list in `order` (best, worst, first, last, "
        "slowest, failed) on `metric` (default: the optimized one), at most `limit` (25), "
        "keeping those matching every `where` condition {field, op, value}: field a "
        "hyperparameter, a metric, phase, status, run, seconds or trial; op one of "
        "= != < <= > >= in."),
    parameters={"type": "object", "properties": {
        "trial": {"type": "integer"},
        "order": {"type": "string", "enum": list(results.ORDERS)},
        "metric": {"type": "string"},
        "limit": {"type": "integer"},
        "where": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string"}, "op": {"type": "string", "enum": list(results.OPERATORS)},
            "value": {}}, "required": ["field", "op"]}},
        "experiment": _EXPERIMENT}},
    handler=_read_trials,
)


def _compare_trials(context, trials: list, experiment: int | None = None) -> ToolResult:
    exp = experiment_of(context, experiment)
    return _answer(exp, _run(lambda: results.compare(trials_of(exp), trials)))


COMPARE_TRIALS = ToolSpec(
    name="compare_trials",
    description=("Two to five trials side by side, by number: every score, their "
                 "configurations, and which hyperparameters they set differently."),
    parameters={"type": "object", "properties": {
        "trials": {"type": "array", "items": {"type": "integer"}}, "experiment": _EXPERIMENT},
        "required": ["trials"]},
    handler=_compare_trials,
)


def _analyze_experiment(context, analysis: str, metric: str = "", hyperparameter: str = "",
                        experiment: int | None = None) -> ToolResult:
    exp = experiment_of(context, experiment)
    return _answer(exp, _run(lambda: results.analyze(trials_of(exp), analysis, metric or None,
                                                     hyperparameter or None)))


ANALYZE_EXPERIMENT = ToolSpec(
    name="analyze_experiment",
    description=(
        "A fixed analysis of the experiment's trials, worked out from every stored trial. "
        "`analysis`: " + "; ".join(f"{k} — {v}" for k, v in results.ANALYSES.items())
        + ". `metric` defaults to the optimized one; `hyperparameter` narrows coverage and "
          "effects to one."),
    parameters={"type": "object", "properties": {
        "analysis": {"type": "string", "enum": list(results.ANALYSES)},
        "metric": {"type": "string"}, "hyperparameter": {"type": "string"},
        "experiment": _EXPERIMENT},
        "required": ["analysis"]},
    handler=_analyze_experiment,
)


#: Where the numbers behind each of the dashboard's figures are read.
FIGURE_DATA = {
    "trials": "read_trials",
    "best_configuration": "read_trials(order=best, limit=1)",
    "selected_configuration": "read_trials(trial=<the selected trial>)",
    "performance_over_time": "analyze_experiment(analysis=progress)",
    "trial_duration": "analyze_experiment(analysis=durations)",
    "hyperparameter_importance": "analyze_experiment(analysis=importance)",
    "interactions_heatmap": "analyze_experiment(analysis=importance)",
    "interactions_top_pairs": "analyze_experiment(analysis=importance)",
    "interactions_graph": "analyze_experiment(analysis=importance)",
    "interactions_coalitions": "analyze_experiment(analysis=importance)",
    "interactions_orders": "analyze_experiment(analysis=importance)",
    "parallel_coordinates": "analyze_experiment(analysis=effects)",
    "partial_dependence": "analyze_experiment(analysis=effects)",
    "local_effects": "analyze_experiment(analysis=effects)",
    "configuration_cube": "analyze_experiment(analysis=coverage)",
    "prior": "read_experiment(aspect=priors)",
}


def experiment_tools():
    """The tools that read an experiment."""
    return [READ_EXPERIMENT, READ_TRIALS, COMPARE_TRIALS, ANALYZE_EXPERIMENT]
