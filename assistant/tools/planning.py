"""`assistant.tools.planning` lists options and writes and checks plans."""

from __future__ import annotations

import json

from .. import plan as plans
from .. import tradeoffs
from .registry import Refused, ToolResult, ToolSpec


def _about(cls) -> str:
    """`_about` returns a class docstring's first paragraph."""
    import inspect

    doc = (cls.__dict__.get("__doc__") or "").strip()
    return " ".join(inspect.cleandoc(doc).split("\n\n")[0].split()) if doc else ""


def _list_models(context, task: str = "") -> ToolResult:
    from core import tasks

    from core.registry import MODELS

    out = []
    for name, model in MODELS.items():
        supported = tasks.supported(model)
        if task and task not in supported:
            continue
        space = model.get_config_space()
        out.append({
            "name": name,
            "tasks": list(supported),
            "forecasts_only": tasks.forecaster(model),
            "accepts_missing_values": getattr(model, "accepts_missing", True),
            "about": _about(type(model)),
            "hyperparameters": [str(h) for h in space.values()],
        })
    return ToolResult(output=json.dumps(out))


LIST_MODELS = ToolSpec(
    name="list_models",
    description=("The models Codesigner can tune: name, the tasks each does, whether it "
                 "only forecasts, whether it takes missing values, and its search space. "
                 "With `task`, only those that do it."),
    parameters={"type": "object",
                "properties": {"task": {"type": "string", "enum": ["classification", "regression"]}}},
    handler=_list_models,
)


def _list_metrics(context, task: str, model: str = "") -> ToolResult:
    from core.metrics import METRICS, metrics_for

    from core.registry import MODELS
    from ui.drafts import free_probabilities

    if model and model not in MODELS:
        raise Refused(f"there is no model called {model!r}; list_models names them")
    probabilities = bool(model) and free_probabilities(task, model)
    scored = metrics_for(task, probabilities=probabilities)
    out = [{"name": name, "higher_is_better": METRICS[name].higher_is_better,
            "range": list(METRICS[name].bounds), "definition": METRICS[name].definition}
           for name in scored]
    left_out = [name for name in metrics_for(task, probabilities=True) if name not in scored]
    note = (f" Every one of these {len(out)} is computed for every trial; the plan's metric "
            "is the one the search optimizes.")
    if left_out:
        note += (f" Left out: {', '.join(left_out)}, which need class probabilities"
                 + (f" that {model} does not give for free." if model
                    else "; name the model to see whether they apply."))
    weighed = tradeoffs.notes(["optimize_metric", "evaluation_scheme"])
    return ToolResult(output=json.dumps(out) + note + " What choosing the metric and the "
                      "validation method trades (`tradeoffs`): "
                      + json.dumps(weighed, ensure_ascii=False))


LIST_METRICS = ToolSpec(
    name="list_metrics",
    description=("The metrics scored for a task with a model: how each is computed, and "
                 "which way is better. Without `model`, those needing class probabilities "
                 "are left out. Also what choosing the metric to optimize and the validation "
                 "method gains and costs, and the circumstances that call for each."),
    parameters={"type": "object",
                "properties": {"task": {"type": "string", "enum": ["classification", "regression"]},
                               "model": {"type": "string",
                                         "description": "One of list_models's names."}},
                "required": ["task"]},
    handler=_list_metrics,
)


def _list_optimizers(context) -> ToolResult:
    from core.registry import OPTIMIZERS
    from ui.optimizer_labels import GROUPS, LABELS

    out = []
    for name, optimizer in OPTIMIZERS.items():
        effective = optimizer.effective_defaults()
        found = tradeoffs.for_optimizer(optimizer)
        out.append({
            "name": name,
            "about": _about(optimizer),
            "settings": [{"name": p.name, "label": p.label,
                          "help": str(LABELS.get(p.name, ("", ""))[1]), "type": p.type,
                          "default": p.default, "blank_means": effective.get(p.name),
                          "min": p.min, "max": p.max, "choices": list(p.choices),
                          "only_when": p.depends_on, "advanced": p.advanced,
                          **({"group": f"{GROUPS[p.group][0]}: {GROUPS[p.group][1]}"}
                             if p.group in GROUPS else {})}
                         for p in optimizer.params_schema],
            **({"tradeoffs": found} if found else {}),
        })
    return ToolResult(output=json.dumps(out, default=str, ensure_ascii=False))


LIST_OPTIMIZERS = ToolSpec(
    name="list_optimizers",
    description=("The optimizers, each with its settings: what each does (`help`), type, "
                 "default, what a blank one runs with (`blank_means`, per search strategy "
                 "where they differ), bounds, choices, and the setting another one depends "
                 "on. A setting left out of the plan keeps its default. `tradeoffs` says, "
                 "for each setting or group, what moving it from its default gains and costs, "
                 "and the circumstances that call for it."),
    parameters={"type": "object", "properties": {}},
    handler=_list_optimizers,
)


def _dataset(context):
    path = context.conversation.dataset_path()
    if path is None:
        raise Refused("there is no dataset yet; ask the person to choose one")
    return path


def _check(context) -> list[str]:
    conversation = context.conversation
    return plans.problems(conversation.plan, context.request(), _dataset(context),
                          conversation.group_id)


def _update_plan(context, **changes) -> ToolResult:
    from ..loop import emit

    _dataset(context)
    conversation = context.conversation
    conversation.plan = plans.merged(conversation.plan, changes)
    conversation.save(update_fields=["plan", "updated_at"])
    found = _check(context)
    emit(conversation, "plan", plan=conversation.plan, problems=found)
    design = _design(conversation.plan)
    if found:
        return ToolResult(output="Plan updated. Not ready yet: " + " ".join(found) + design)
    return ToolResult(output="Plan updated." + design + " It can be set up as it stands. "
                             "Stop here: tell the "
                             "person in a sentence or two what the plan is, and that they "
                             "approve it with the button under it or say what to change. "
                             "There is nothing to open or fill yet.")


UPDATE_PLAN = ToolSpec(
    name="update_plan",
    description=(
        "Write down what has been decided, as soon as it is decided: only the fields that "
        "change; a section is merged field by field. The person sees the plan as it "
        "stands, and what still stops it being set up."),
    parameters=plans.SCHEMA,
    handler=_update_plan,
)


def _design(plan: dict) -> str:
    sentence = plans.design(plan)
    return f" {sentence}" if sentence else ""


def _check_plan(context) -> ToolResult:
    found = _check(context)
    design = _design(context.conversation.plan)
    if found:
        return ToolResult(output="Not ready: " + " ".join(found) + design)
    return ToolResult(output="The plan can be set up as it stands." + design
                             + " Stop here: tell the person "
                             "it is ready and that they approve it with the button under it.")


CHECK_PLAN = ToolSpec(
    name="check_plan",
    description=("Check the plan against the setup's own forms, saving nothing: what would "
                 "stop it being set up, or that it can be."),
    parameters={"type": "object", "properties": {}},
    handler=_check_plan,
)


def lookups():
    """`lookups` lists catalogue tools."""
    return [LIST_MODELS, LIST_METRICS, LIST_OPTIMIZERS]


def planning():
    """`planning` adds plan tools to `lookups`."""
    return lookups() + [UPDATE_PLAN, CHECK_PLAN]
