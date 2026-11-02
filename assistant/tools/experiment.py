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

from .. import results, tradeoffs
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


def _runs(exp, stored: int) -> list[dict]:
    """*exp*'s runs, each with its trial range. A run still going has no count
    yet, so its range runs to the *stored* trials past its offset."""
    out = []
    for index, run in enumerate(exp.runs.order_by("pk"), start=1):
        row = {"index": index, "status": run.status, "stopped_by": run.stopped_by or None,
               "stopping": run.stopping or {}, "first": None, "last": None}
        if run.trial_offset is not None and run.trial_count is not None:
            row["first"], row["last"] = run.trial_offset + 1, run.trial_offset + run.trial_count
        elif (run.trial_offset is not None and run.status in results.ACTIVE
              and stored > run.trial_offset):
            row["first"], row["last"] = run.trial_offset + 1, stored
        out.append(row)
    return out


def _stored(exp) -> int:
    """How many trials *exp* stores now, which a running run raises with each
    progress write."""
    return len((exp.data.result or {}).get("data") or [])


def _planned(exp, kind, space) -> tuple[dict | None, int | None]:
    """The initial design SMAC sizes for *exp*'s latest run, and that run's
    whole budget, from the settings it started with; None for another
    optimizer or without a run or space."""
    from core.optimizers.smac.facade import initial_points, scenario_budget

    latest = exp.runs.order_by("pk").last()
    if latest is None or space is None or getattr(kind, "name", "") != "SMAC":
        return None, None
    budget = scenario_budget(latest.trial_offset or 0, latest.max_trials)
    params = latest.optimizer_params or exp.data.optimizer_params or {}
    size = initial_points(len(list(space.values())), budget, params)
    return {"size": size.trials, "bound_by": size.bound_by, "budget": budget}, budget


def trials_of(exp) -> results.Trials:
    """*exp*'s trials, rebuilt from what it stores; kept until a run or the
    stored result changes."""
    from experiments.services.snapshot import config_space_for, rebuild_experiment

    stored = _stored(exp)
    runs = _runs(exp, stored)
    key = (exp.pk, exp.data_id, stored, len(runs), tuple((r["status"], r["last"]) for r in runs),
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
    space = config_space_for(built)
    planned, budget = _planned(exp, kind, space)
    found = results.Trials(result=built["result"], space=space,
                           optimizer=kind.name if kind is not None else exp.data.optimizer_name,
                           settings=dict(exp.data.optimizer_params or {}), runs=runs,
                           planned=planned, budget=budget)
    with _kept_lock:
        _KEPT[key] = found
        while len(_KEPT) > _KEPT_SIZE:
            _KEPT.popitem(last=False)
    return found


def _answer(exp, found: dict, t: results.Trials) -> ToolResult:
    """*found* as the tool's output, stamped with the trials and run status it
    was worked out at, since a running run makes it out of date."""
    stamp = {"trials": len(t.trials)}
    if t.runs:
        stamp["run"] = {k: t.runs[-1][k] for k in ("index", "status")}
    if t.live is not None:
        stamp["note"] = ("a run is still adding trials, so this is out of date as soon as "
                         "the next one lands; ask again before relying on it later")
    return ToolResult(output=json.dumps({"as_of": stamp, **found}, default=str,
                                        ensure_ascii=False),
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
    out["trials_by_phase"] = t.phase_counts()
    if t.design:
        design_done = out["trials_by_phase"].get(results.DESIGN, 0)
        out["initial_design"] = {"planned": t.design["size"], "done": design_done,
                                 "bound_by": t.design.get("bound_by"),
                                 "planned_from": t.design["source"]}
    if t.runs:
        out["last_run"] = {k: t.runs[-1][k] for k in ("index", "status", "stopped_by",
                                                       "first", "last")}
    if t.live is not None:
        out["running"] = True
    if data.trial_budget:
        out["trial_budget"] = {"planned": data.trial_budget, "done": len(t.trials)}
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
    data = exp.data
    space = []
    for hp in t.hyperparameters():
        row = {"name": hp.name, "type": type(hp).__name__,
               "default": getattr(hp, "default_value", None)}
        if hasattr(hp, "lower"):
            row.update({"lower": hp.lower, "upper": hp.upper, "log": bool(hp.log)})
        if hasattr(hp, "choices"):
            row["choices"] = list(hp.choices)
        space.append(row)
    settings = _optimizer_settings(exp, t)
    out = {"optimizer": data.optimizer_name, "optimizer_settings": settings}
    from core.registry import optimizer_class

    kind = optimizer_class(data.optimizer_name)
    editable = [name for name, row in settings.items() if row["editable_now"]]
    # The metric is chosen for each run, so its note always applies.
    out["tradeoffs"] = {**(tradeoffs.for_optimizer(kind, editable) if kind else {}),
                        **tradeoffs.notes(["optimize_metric"])}
    if t.design:
        out["initial_design_these_give"] = {
            "planned": t.design["size"], "done": t.phase_counts().get(results.DESIGN, 0),
            "bound_by": t.design.get("bound_by"), "planned_from": t.design["source"]}
    return {**out, "search_space": space, "seed": data.seed, "cv_folds": data.cv_folds or None,
            "test_size": data.test_size if not data.cv_folds else None,
            "time_column": data.time_column or None, "processing": data.processing or {}}


def _design_grows_from(n_hyperparameters, budget, params, drawn) -> int | None:
    """The smallest total of trials above *budget* at which the initial design
    holds more than the *drawn* points, or None when the caps keep it at most
    that at any total. The design's size never shrinks as the total grows."""
    from core.optimizers.smac.facade import initial_points

    def bigger(total):
        return initial_points(n_hyperparameters, total, params).trials > drawn

    low, high = budget, 100_000
    if bigger(low) or not bigger(high):
        return None
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if bigger(middle) else (middle, high)
    return high


def _optimizer_settings(exp, t) -> dict:
    """Each optimizer setting that applies under the stored search strategy: the
    stored value (blank meaning its default), the value in effect, and whether
    it can change now, with why not and where it changes."""
    from core.optimizers.smac.params import signature_default

    from core.registry import optimizer_class
    from ui.optimizer_form import has_trials, why_fixed
    from ui.optimizer_labels import LABELS

    kind = optimizer_class(exp.data.optimizer_name)
    if kind is None:
        return {}
    stored = kind.known_params(exp.data.optimizer_params or {})
    try:
        in_effect = kind(**stored).resolved_params()
    except (TypeError, ValueError):
        in_effect = {}
    effective = kind.effective_defaults()
    strategy = stored.get("search_strategy")
    where = ("the run form's Next run fold on the experiment's page, saved when the run "
             "starts" if has_trials(exp) else "the optimizer form")
    out = {}
    for param in kind.params_schema:
        if (param.depends_on and param.depends_on[0] == "search_strategy" and strategy
                and strategy != param.depends_on[1]):
            continue
        value = stored.get(param.name)
        current = in_effect.get(param.name)
        if current is None:
            current = effective.get(param.name)
        if param.name == "trial_cap" and value is None and t.space is not None:
            from smac.initial_design import AbstractInitialDesign

            per_hp = signature_default(AbstractInitialDesign.__init__,
                                       "n_configs_per_hyperparameter") or 10
            current = f"{per_hp * len(t.hp_names())} ({per_hp} per hyperparameter)"
        row = {"label": str(LABELS.get(param.name, (param.label, ""))[0]),
               "stored": "blank, so its default" if value is None else value,
               "in_effect": current}
        fixed = why_fixed(exp, param.name)
        row["editable_now"] = fixed is None
        row["changes_in" if fixed is None else "why_not"] = where if fixed is None else fixed
        out[param.name] = row
    return out


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


def _form_values(context, exp) -> tuple[dict, str]:
    """The run form's fields, as the person's page shows them when it is this
    experiment's, else as the form opens; and which of the two."""
    from .. import acts

    page = acts.current(context.conversation)
    if (page is not None and page.experiment_id == exp.pk
            and page.url_name == "ui:experiment_detail"):
        values = {f["name"]: f.get("value") for f in page.fields or [] if f.get("name")}
        if "max_trials" in values:
            return values, "the run form as the person's page shows it now"
    from ui.run_form import opening_trials

    values = {"max_trials": opening_trials(exp.data.trial_budget, _stored(exp)),
              "max_failures": 1, "max_consecutive_failures": 15,
              "optimize_metric": exp.data.current_metric or ""}
    return values, "the run form as it opens, since the person's page does not show it"


def _stops(stopping: dict, best, higher_is_better: bool, done: int, budget) -> list[str]:
    """Each criterion in *stopping* as what it will do to the run."""
    out = []
    for key, value in stopping.items():
        if key == "max_trials":
            line = f"max_trials {value}: stops after {value} more trials, {done + value} in all"
            if budget:
                line += (f"; the plan's budget is {budget}, so this "
                         + ("finishes it" if done + value == budget else
                            f"{'overshoots' if done + value > budget else 'leaves'} "
                            f"{abs(budget - done - value)}"))
            out.append(line + ".")
        elif key == "target_score":
            close = best is not None and (value - best if higher_is_better else best - value) < 1e-3
            out.append(f"target_score {value}: " + (
                f"stops at the first trial that beats the best so far ({results._r(best)}), however "
                "slightly" if close else f"stops once a trial scores {value} or better") + ".")
        elif key == "max_failures":
            out.append(f"max_failures {value}: " + (
                "stops at the first failed trial" if value == 1
                else f"stops after {value} failed trials") + ".")
        elif key == "max_consecutive_failures":
            out.append(f"max_consecutive_failures {value}: stops after {value} failures in a row.")
        elif key == "no_improvement_trials":
            out.append(f"no_improvement_trials {value}: stops after {value} trials in a row "
                       "without a new best.")
        else:
            out.append(f"{key} {value}.")
    return out


def _next_run(context, exp, t, trials: int | None = None) -> dict:
    """What pressing Run would do with the run form as it stands, or with
    *trials* in place of its trial limit."""
    from core.optimizers.smac.facade import initial_points, scenario_budget
    from core.optimizers.smac.params import EXTENDABLE_DESIGNS

    from core.registry import optimizer_class
    from ui import run_form
    from ui.optimizer_form import cleaned_params, why_fixed

    if t.live is not None:
        return {"can_run": False, "why": "a run is under way; the next can start when it ends"}
    values, source = _form_values(context, exp)
    if trials:
        values = {**values, "max_trials": trials}
        source += f", with a trial limit of {trials} asked about instead of the form's"
    metric = values.get("optimize_metric") or exp.data.current_metric or ""
    stopping = run_form.stopping_from(values, metric)
    done = len(t.trials)
    out = {"read_from": source, "trials_so_far": done, "metric": metric,
           "metric_note": "The metric is chosen for each run, so this run may optimize another."}
    kind = optimizer_class(exp.data.optimizer_name)
    stored = kind.known_params(exp.data.optimizer_params or {}) if kind else {}
    filled = {k[4:]: v for k, v in values.items() if k.startswith("opt_")}
    params = (cleaned_params(kind, {**stored, **{k: v for k, v in filled.items()
                                                 if k in kind.run_safe}})
              if kind is not None else {})
    best = None
    if t.trials and metric in {m for x in t.trials for m in x.scores}:
        ranked = t.ranked(metric)
        best = t.score(ranked[0], metric) if ranked else None
    higher = t.higher_is_better(metric) if t.trials and best is not None else True
    out["stops"] = _stops(stopping, best, higher, done, exp.data.trial_budget)
    if not any(k not in ("max_failures", "max_consecutive_failures") for k in stopping):
        out["stops"].append("Nothing but the failure limits would end it: the run form "
                            "refuses that.")
    max_trials = stopping.get("max_trials")
    if max_trials:
        out["trials_this_run"] = max_trials
    if kind is not None and kind.name == "SMAC" and t.space is not None:
        budget = scenario_budget(done, max_trials)
        size = initial_points(len(t.hp_names()), budget, params)
        design_done = t.phase_counts().get(results.DESIGN, 0)
        extendable = params.get("initial_design", "sobol") in EXTENDABLE_DESIGNS
        draws = max(0, size.trials - design_done) if extendable else 0
        out["budget_for_sizing"] = budget
        none_left = f"No design points are left to draw at {budget} trials in all; the model " \
                    f"chooses every trial."
        grows = _design_grows_from(len(t.hp_names()), budget, params, design_done)
        if grows is not None:
            none_left += (f" A run taking the total to {grows} or more re-sizes the design "
                          f"and draws more points first.")
        out["initial_design"] = {
            "size_at_this_budget": size.trials, "bound_by": size.bound_by,
            "already_drawn": design_done, "will_draw": draws,
            "reads": (f"The design is re-sized to the larger budget, so this run first draws "
                      f"{draws} more design points, then the model chooses." if draws else
                      none_left)
            if extendable else "A Latin hypercube design keeps its first size on a resume."}
        if not max_trials:
            out["initial_design"]["note"] = (f"With no trial limit the design is sized as if "
                                             f"for {budget} trials in all.")
        rate = params.get("random_probability")
        effective = kind.effective_defaults().get("random_probability")
        if rate is None and isinstance(effective, dict):
            rate = effective.get(params.get("search_strategy", "gp"))
        out["random_trial_rate"] = rate if rate is not None else effective
    out["can_change_before_it"] = (
        ["the run form's limits and metric"]
        + [f"opt_{name}" for name in sorted(getattr(kind, "run_safe", ()))
           if why_fixed(exp, name) is None])
    out["fixed"] = "every other optimizer setting, the model, the space and the evaluation"
    return out


ASPECTS = {"overview": _overview, "runs": _runs_aspect, "settings": _settings,
           "priors": _priors, "timeline": _timeline, "next_run": None}


def _read_experiment(context, aspect: str = "overview", experiment: int | None = None,
                     trials: int | None = None) -> ToolResult:
    exp = experiment_of(context, experiment)
    t = trials_of(exp)
    found = (_next_run(context, exp, t, trials) if aspect == "next_run"
             else ASPECTS[aspect](exp, t))
    return _answer(exp, found, t)


_EXPERIMENT = {"type": "integer",
               "description": "Which experiment; by default the conversation's, else the one "
                              "whose page is open."}

READ_EXPERIMENT = ToolSpec(
    name="read_experiment",
    description=(
        "What an experiment is, from what it stores. `aspect`: overview (task, data, model, "
        "optimizer, metric, trial and failure counts, trials per phase, the best trial, the "
        "initial design planned and done, whether a run is going), runs (each run's "
        "trials, how it was set to stop and why it stopped), "
        "settings (each optimizer setting stored and in effect, whether it can change "
        "now, for those that can and for the metric when moving them from the default is "
        "worth it and what it costs, and the search space with bounds), priors (each stated prior and how much it "
        "still counts), timeline (what was done, when, by whom), next_run (what pressing Run "
        "would do with the run form as filled: trials, the new budget, how many initial-"
        "design points it draws, the random rate, what each early stop means, and what can "
        "still change before it; `trials` asks about a run of that many instead of the "
        "form's limit). Every answer says the trials and run status it was made at."),
    parameters={"type": "object", "properties": {
        "aspect": {"type": "string", "enum": list(ASPECTS)}, "experiment": _EXPERIMENT,
        "trials": {"type": "integer",
                   "description": "With next_run: a trial limit to ask about."}}},
    handler=_read_experiment,
)


# ── read_trials, compare_trials, analyze_experiment ─────────────────────────

def _read_trials(context, trial: int | None = None, order: str = "best",
                 metric: str = "", limit: int = 10, where: list | None = None,
                 experiment: int | None = None) -> ToolResult:
    exp = experiment_of(context, experiment)
    t = trials_of(exp)
    if trial is not None:
        return _answer(exp, _run(lambda: results.trial(t, int(trial), metric or None)), t)
    return _answer(exp, _run(lambda: results.rows(t, order, metric or None, limit, where)), t)


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
    t = trials_of(exp)
    return _answer(exp, _run(lambda: results.compare(t, trials)), t)


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
    t = trials_of(exp)
    return _answer(exp, _run(lambda: results.analyze(t, analysis, metric or None,
                                                     hyperparameter or None)), t)


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


# ── list_experiments ────────────────────────────────────────────────────────

#: The most experiments one list gives.
LISTED = 50


def _list_experiments(context) -> ToolResult:
    from access.policy import visible_experiments

    out = []
    found = (visible_experiments(context.request()).select_related("data")
             .order_by("-created_at")[:LISTED])
    for exp in found:
        data = exp.data
        latest = exp.runs.order_by("pk").last()
        out.append({"experiment": exp.pk, "name": exp.title,
                    "kind": "draft" if exp.draft else "created",
                    "task": data.task, "model": data.model_name,
                    "dataset": data.demo_dataset or (data.dataset.name.rsplit("/", 1)[-1]
                                                     if data.dataset else None),
                    "target": data.target_column or None, "optimizer": data.optimizer_name,
                    "trials": _stored(exp),
                    "last_run": latest.status if latest is not None else None})
    return ToolResult(output=json.dumps({"experiments": out, "shown": len(out)}, default=str))


LIST_EXPERIMENTS = ToolSpec(
    name="list_experiments",
    description=(
        "The experiments the person can see, newest first: each one's id, name, whether it "
        "is a draft, task, model, dataset, target, optimizer, trials stored and the last "
        "run's status. Use it to find related experiments; read_experiment with "
        "`experiment` reads any of them."),
    parameters={"type": "object", "properties": {}},
    handler=_list_experiments,
)


def experiment_tools():
    """The tools that read an experiment, and the one that lists them."""
    return [LIST_EXPERIMENTS, READ_EXPERIMENT, READ_TRIALS, COMPARE_TRIALS, ANALYZE_EXPERIMENT]
