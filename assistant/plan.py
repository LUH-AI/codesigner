"""`plan` describes the agreed experiment.
"""

from __future__ import annotations

from ui.forms import EVALUATION_BACKTEST, EVALUATION_HOLDOUT, EVALUATION_KFOLD

#: `SCHEMA` is the `update_plan` argument schema.
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
                "max_failures": {"type": "integer",
                                 "description": "Give up after this many failed trials in "
                                                "all; the run form's default is 1."},
                "max_consecutive_failures": {"type": "integer",
                                             "description": "Give up after this many failed "
                                                            "in a row; the default is 15."},
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

REQUIRED = ("task", "model_name", "evaluation", "optimizer")


def merged(plan: dict, changes: dict) -> dict:
    """`merged` applies *changes* over *plan*, merging each section's inner
    dict."""
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
    """`setup_data` builds step 1's form data."""
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
    optimizer = plan.get("optimizer") or {}
    data = {"optimizer_name": optimizer.get("name") or ""}
    for name, value in (optimizer.get("params") or {}).items():
        data[f"opt_{name}"] = value
    return data


def processing_data(plan: dict) -> dict:
    return dict(plan.get("processing") or {})


RUN = "run"


def run_items(plan: dict) -> list[tuple]:
    """`run_items` returns the run form's fields. Stopping without `max_trials`
    clears the trial limit."""
    from ui.run_form import STOPPING_FIELDS

    items = [("optimize_metric", plan["metric"])] if plan.get("metric") else []
    stopping = plan.get("stopping") or {}
    if stopping:
        items.append(("max_trials", stopping.get("max_trials", "")))
        items += [(k, stopping[k]) for k in STOPPING_FIELDS
                  if k != "max_trials" and stopping.get(k) not in (None, "")]
    return items


#: `_FROM` maps form fields to plan sections.
_FROM = {"evaluation_scheme": "evaluation", "evaluation_value": "evaluation",
         "time_column": "evaluation", "time_gap": "evaluation", "horizon": "evaluation",
         "series_column": "evaluation", "season": "evaluation", "optimizer_name": "optimizer",
         "missing": "processing", "scale": "processing", "labels": "processing",
         "optimize_metric": "metric", "max_trials": "stopping", "target_score": "stopping",
         "max_seconds": "stopping", "max_trial_seconds": "stopping",
         "no_improvement_trials": "stopping", "incumbent_confidence": "stopping",
         "max_failures": "stopping", "max_consecutive_failures": "stopping"}


def fill_for(plan: dict, step: str) -> list[dict]:
    """`fill_for` returns the value and reason of each *step* field *plan* sets."""
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
    """`problems` returns what keeps *plan* from being set up on *dataset*:
    missing keys or data, the setup forms' errors, an unknown optimizer or bad
    optimizer parameters, and processing, stopping or metric choices the model
    and task cannot take. It saves nothing."""
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
        found += _optimizer_problems(chosen, optimizer.get("params") or {})

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

        from ui.drafts import free_probabilities

        task = plan.get("task") or ""
        probabilities = model is not None and free_probabilities(task, plan["model_name"])
        if metric not in metrics_for(task, probabilities=probabilities):
            found.append(f"metric: {metric!r} is not scored for a {task} with "
                         f"{plan.get('model_name')}.")
    return found


def _optimizer_problems(optimizer, params: dict) -> list[str]:
    """`_optimizer_problems` reports settings in *params* that are unknown,
    altered by the optimizer form, or inactive."""
    from ui.optimizer_form import cleaned_params

    schema = {param.name: param for param in optimizer.params_schema}
    found = [f"optimizer: {optimizer.name} has no setting {name!r}." for name in params
             if name not in schema]
    given = {k: v for k, v in params.items() if k in schema}
    stored = cleaned_params(optimizer, given)
    for name, value in given.items():
        if value is None:
            continue
        kept = stored[name]
        if kept != value and str(kept) != str(value):
            found.append(f"optimizer: {name} {value!r} would be stored as {kept!r}"
                         + _range(schema[name]) + ".")
            continue
        param = schema[name]
        if param.depends_on and stored.get(param.depends_on[0]) != param.depends_on[1]:
            found.append(f"optimizer: {name} applies only when {param.depends_on[0]} is "
                         f"{param.depends_on[1]!r}.")
        for flag in param.enabled_by or ():
            if not stored.get(flag):
                found.append(f"optimizer: {name} does nothing while {flag} is off.")
    return found


def _range(param) -> str:
    if param.choices:
        return f"; it takes one of {list(param.choices)}"
    if param.min is not None or param.max is not None:
        return f"; it runs from {param.min} to {param.max}"
    return ""


def design(plan: dict) -> str:
    """`design` describes *plan*'s SMAC initial design, else returns ""."""
    from core.optimizers.smac.facade import initial_points, scenario_budget

    from core.registry import MODELS

    optimizer = plan.get("optimizer") or {}
    model = MODELS.get(plan.get("model_name"))
    if optimizer.get("name") != "SMAC" or model is None:
        return ""
    stopping = plan.get("stopping") or {}
    budget = scenario_budget(0, stopping.get("max_trials"))
    size = initial_points(len(list(model.get_config_space().values())), budget,
                          optimizer.get("params") or {})
    of = (f"{budget} trials" if stopping.get("max_trials")
          else f"the {budget} assumed when no trial limit is set")
    return (f"Initial design: {size.trials} of {of} ({size.bound_by}); the model chooses "
            f"the rest.")
