"""The plan: what the person and the assistant agree an experiment will be,
field for field what its setup asks.

Kept on the conversation (`Conversation.plan`) as it is worked out, checked
against the setup's own forms without saving anything (`problems`), and
approved by the person before the setup is filled from it.
"""

from __future__ import annotations

from ui.forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD

#: The plan's fields, as the model may state them (`update_plan`). Each maps to
#: a field of the setup or of the first run's form.
SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "description": "What the experiment is for, in a sentence."},
        "name": {"type": "string", "description": "A short name for the experiment."},
        "task": {"type": "string", "enum": ["classification", "regression"]},
        "target_column": {"type": "string",
                          "description": "The column to predict; empty for the last."},
        "model_name": {"type": "string", "description": "One of list_models's names."},
        "evaluation": {
            "type": "object",
            "properties": {
                "scheme": {"type": "string",
                           "enum": [EVALUATION_KFOLD, EVALUATION_HOLDOUT, EVALUATION_BACKTEST]},
                "value": {"type": "number",
                          "description": "Folds for kfold, share held out for holdout, "
                                         "backtests for backtest."},
                "time_column": {"type": "string"},
                "time_gap": {"type": "integer"},
                "horizon": {"type": "integer"},
                "series_column": {"type": "string"},
                "season": {"type": "integer"},
            },
        },
        "optimizer": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "One of list_optimizers's names."},
                "params": {"type": "object", "additionalProperties": True,
                           "description": "Settings by name, as list_optimizers gives them; "
                                          "leave out what should stay at its default."},
            },
        },
        "processing": {
            "type": "object",
            "properties": {
                "missing": {"type": "string", "enum": ["auto", "impute", "keep"]},
                "scale": {"type": "string", "enum": ["auto", "standardize", "none"]},
                "labels": {"type": "string", "enum": ["auto", "one_hot", "categories"]},
            },
        },
        "metric": {"type": "string", "description": "The metric to optimize, from list_metrics."},
        "stopping": {
            "type": "object",
            "properties": {
                "max_trials": {"type": "integer"},
                "max_seconds": {"type": "number"},
                "no_improvement_trials": {"type": "integer"},
                "target_score": {"type": "number"},
            },
        },
        "seed": {"type": "integer"},
        "reasons": {
            "type": "object", "additionalProperties": {"type": "string"},
            "description": "Why, in one short sentence each, keyed by the field it is for "
                           "(task, model_name, evaluation, optimizer, processing, metric, "
                           "stopping, ...). Shown beside the field it filled."},
    },
}

#: What the setup cannot be filled from without.
REQUIRED = ("task", "model_name", "evaluation", "optimizer")


def merged(plan: dict, changes: dict) -> dict:
    """*plan* with *changes* laid over it, nested sections field by field."""
    out = dict(plan or {})
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            section = dict(out[key])
            for inner, inner_value in value.items():
                if isinstance(inner_value, dict) and isinstance(section.get(inner), dict):
                    section[inner] = {**section[inner], **inner_value}
                else:
                    section[inner] = inner_value
            out[key] = section
        else:
            out[key] = value
    return out


def setup_data(plan: dict, group_id=None) -> dict:
    """Step 1's form, as the page would post it, from *plan* — the dataset in
    hand kept, the experiment in group *group_id*."""
    evaluation = plan.get("evaluation") or {}
    scheme = evaluation.get("scheme") or EVALUATION_KFOLD
    default = {EVALUATION_KFOLD: 5, EVALUATION_HOLDOUT: 0.2, EVALUATION_BACKTEST: 3}[scheme]
    data = {
        "name": plan.get("name") or "",
        "task": plan.get("task") or "",
        "model_name": plan.get("model_name") or "",
        "seed": plan.get("seed", 0),
        "evaluation_scheme": scheme,
        "evaluation_value": evaluation.get("value", default),
        "time_column": evaluation.get("time_column") or "",
        "time_gap": evaluation.get("time_gap") or 0,
        "horizon": evaluation.get("horizon") or (12 if scheme == EVALUATION_BACKTEST else ""),
        "series_column": evaluation.get("series_column") or "",
        "season": evaluation.get("season") or "",
        "target_column": plan.get("target_column") or "",
    }
    if group_id:
        data["group"] = group_id
    return data


def optimizer_data(plan: dict) -> dict:
    """Step 2's form from *plan*: the optimizer and the settings it names."""
    optimizer = plan.get("optimizer") or {}
    data = {"optimizer_name": optimizer.get("name") or ""}
    for name, value in (optimizer.get("params") or {}).items():
        data[f"opt_{name}"] = value
    return data


def processing_data(plan: dict) -> dict:
    """Step 3's form from *plan*."""
    return dict(plan.get("processing") or {})


#: The run form on a created experiment's page, filled like a setup step.
RUN = "run"


def run_items(plan: dict) -> list[tuple]:
    """The run form's fields from *plan*: the metric, and the stopping criteria
    it states. The form's default trial limit is cleared when the plan does
    not keep one, so the run ends on what was agreed."""
    from ui.run_form import STOPPING_FIELDS

    items = [("optimize_metric", plan["metric"])] if plan.get("metric") else []
    stopping = plan.get("stopping") or {}
    if stopping:
        items.append(("max_trials", stopping.get("max_trials", "")))
        items += [(k, stopping[k]) for k in STOPPING_FIELDS
                  if k != "max_trials" and stopping.get(k) not in (None, "")]
    return items


#: Which of the plan's fields each form field is filled from, for its reason.
_FROM = {"evaluation_scheme": "evaluation", "evaluation_value": "evaluation",
         "time_column": "evaluation", "time_gap": "evaluation", "horizon": "evaluation",
         "series_column": "evaluation", "season": "evaluation", "optimizer_name": "optimizer",
         "missing": "processing", "scale": "processing", "labels": "processing",
         "optimize_metric": "metric", "max_trials": "stopping", "target_score": "stopping",
         "max_seconds": "stopping", "max_trial_seconds": "stopping",
         "no_improvement_trials": "stopping", "incumbent_confidence": "stopping",
         "max_failures": "stopping", "max_consecutive_failures": "stopping"}


def fill_for(plan: dict, step: str) -> list[dict]:
    """What setup *step*'s page is filled with from *plan*: each field's name,
    value and the plan's reason for it, in the order the page needs them (a
    field that decides what the next offers comes first). Only what the plan
    says; the rest of the page is left as it is."""
    from ui import setup_steps as steps

    if step == steps.SETUP:
        data = setup_data(plan)
        evaluation = plan.get("evaluation") or {}
        order = ["name", "task", "model_name", "target_column", "evaluation_scheme",
                 "evaluation_value", "time_column", "time_gap", "horizon", "series_column",
                 "season", "seed"]

        def stated(name):
            if _FROM.get(name) == "evaluation":
                return bool(evaluation) and (name in ("evaluation_scheme", "evaluation_value")
                                             or name in evaluation)
            return name in plan

        items = [(k, data[k]) for k in order if stated(k) and data.get(k) not in ("", None)]
    elif step == steps.OPTIMIZER:
        data = optimizer_data(plan)
        items = [(k, v) for k, v in data.items() if v not in ("", None)]
    elif step == steps.DATA:
        items = list(processing_data(plan).items())
    elif step == RUN:
        items = run_items(plan)
    else:
        items = []
    reasons = plan.get("reasons") or {}
    out = []
    for name, value in items:
        source = "optimizer" if name.startswith("opt_") else _FROM.get(name, name)
        out.append({"name": name, "value": value, "reason": str(reasons.get(source) or "")})
    return out


def problems(plan: dict, request, dataset, group_id=None) -> list[str]:
    """What stops *plan* being set up on the dataset at *dataset* (a path) in
    group *group_id*, in the words the setup's own forms would use — or
    nothing. Nothing is saved, and no draft is needed: this is what is checked
    before the person approves the plan, and again when they do."""
    from core.processing import CHOICES, unavailable

    from core.registry import MODELS, OPTIMIZERS
    from ui import drafts, run_form

    found = [f"The plan has no {key}." for key in REQUIRED if not plan.get(key)]
    if dataset is None:
        found.append("There is no dataset yet.")
    if found:
        return found

    form = drafts.setup_form_for(request, setup_data(plan, group_id), {}, current_dataset=dataset)
    if not form.is_valid():
        for field, errors in form.errors.get_json_data().items():
            for error in errors:
                found.append(f"{'' if field == '__all__' else field + ': '}{error['message']}")

    optimizer = plan.get("optimizer") or {}
    chosen = OPTIMIZERS.get(optimizer.get("name"))
    if chosen is None:
        found.append(f"optimizer: there is no optimizer called {optimizer.get('name')!r}; "
                     f"there are {sorted(OPTIMIZERS)}.")
    else:
        known = {param.name for param in chosen.params_schema}
        for name in optimizer.get("params") or {}:
            if name not in known:
                found.append(f"optimizer: {chosen.name} has no setting {name!r}.")

    model = MODELS.get(plan.get("model_name"))
    for step, choice in (plan.get("processing") or {}).items():
        if step in CHOICES and model is not None and unavailable(model, step, choice):
            found.append(f"processing: {plan['model_name']} cannot take {choice!r} for {step}.")

    stopping = plan.get("stopping") or {}
    if stopping and not run_form.stopping_from(stopping, plan.get("metric") or ""):
        found.append("stopping: none of these would end a run.")
    metric = plan.get("metric")
    if metric:
        from core.metrics import metrics_for

        if metric not in metrics_for(plan.get("task") or ""):
            found.append(f"metric: {metric!r} is not scored for a {plan.get('task')}.")
    return found
