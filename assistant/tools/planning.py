"""The tools for working out a plan with the person: what Codesigner offers,
and writing the plan down and checking it."""

from __future__ import annotations

import json

from .. import plan as plans
from .registry import Refused, ToolResult, ToolSpec


def _list_models(context, task: str = "") -> ToolResult:
    from core import tasks

    from ui.registry import MODELS

    out = []
    for name, model in MODELS.items():
        supported = tasks.supported(model)
        if task and task not in supported:
            continue
        doc = (type(model).__doc__ or "").strip().splitlines()
        space = model.get_config_space()
        out.append({
            "name": name,
            "tasks": list(supported),
            "forecasts_only": tasks.forecaster(model),
            "accepts_missing_values": getattr(model, "accepts_missing", True),
            "about": doc[0] if doc else "",
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


def _list_metrics(context, task: str) -> ToolResult:
    from core.metrics import METRICS, metrics_for

    out = [{"name": name, "higher_is_better": METRICS[name].higher_is_better,
            "range": list(METRICS[name].bounds)} for name in metrics_for(task)]
    return ToolResult(output=json.dumps(out)
                      + " Every one of these is computed for every trial; the plan's metric "
                        "is the one the search optimizes.")


LIST_METRICS = ToolSpec(
    name="list_metrics",
    description="The metrics scored for a task, and which way is better.",
    parameters={"type": "object",
                "properties": {"task": {"type": "string", "enum": ["classification", "regression"]}},
                "required": ["task"]},
    handler=_list_metrics,
)


def _list_optimizers(context) -> ToolResult:
    from ui.registry import OPTIMIZERS

    out = []
    for name, optimizer in OPTIMIZERS.items():
        out.append({
            "name": name,
            "about": ((type(optimizer).__doc__ or "").strip().splitlines() or [""])[0],
            "settings": [{"name": p.name, "label": p.label, "type": p.type,
                          "default": p.default, "min": p.min, "max": p.max,
                          "choices": list(p.choices), "only_when": p.depends_on,
                          "advanced": p.advanced}
                         for p in optimizer.params_schema],
        })
    return ToolResult(output=json.dumps(out, default=str))


LIST_OPTIMIZERS = ToolSpec(
    name="list_optimizers",
    description=("The optimizers, each with its settings: type, default, bounds, choices, "
                 "and the setting another one depends on. A setting left out of the plan "
                 "keeps its default."),
    parameters={"type": "object", "properties": {}},
    handler=_list_optimizers,
)


def _dataset(context):
    """The dataset this conversation plans for."""
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
    if found:
        return ToolResult(output="Plan updated. Not ready yet: " + " ".join(found))
    return ToolResult(output="Plan updated. It can be set up as it stands; the person "
                             "approves it with the button under it.")


UPDATE_PLAN = ToolSpec(
    name="update_plan",
    description=(
        "Write down what has been decided, as soon as it is decided: only the fields that "
        "change; a section is merged field by field. The person sees the plan as it "
        "stands, and what still stops it being set up."),
    parameters=plans.SCHEMA,
    handler=_update_plan,
)


def _check_plan(context) -> ToolResult:
    found = _check(context)
    if found:
        return ToolResult(output="Not ready: " + " ".join(found))
    return ToolResult(output="The plan can be set up as it stands.")


CHECK_PLAN = ToolSpec(
    name="check_plan",
    description=("Check the plan against the setup's own forms, saving nothing: what would "
                 "stop it being set up, or that it can be."),
    parameters={"type": "object", "properties": {}},
    handler=_check_plan,
)


def lookups():
    """What Codesigner offers: models, metrics, optimizers."""
    return [LIST_MODELS, LIST_METRICS, LIST_OPTIMIZERS]


def planning():
    """The tools for planning, beside the ones every conversation has."""
    return lookups() + [UPDATE_PLAN, CHECK_PLAN]
