"""The tools every conversation has: looking at the dataset, asking the person
something, and saying how to do what only the person may do."""

from __future__ import annotations

from ui.permissions import BOTH, VIEW

from .registry import Refused, ToolResult, ToolSpec


def _profile_dataset(context, demo: str = "") -> ToolResult:
    import json

    from core import io

    from ..profile import profile

    experiment_id = None
    if demo:
        path = io.demo_datasets().get(demo)
        if path is None:
            raise Refused(f"there is no demo dataset called {demo!r}; there are "
                          f"{sorted(io.demo_datasets())}")
        target = ""
    elif context.conversation.dataset_path() is not None:
        path, target = context.conversation.dataset_path(), ""
    else:
        if context.conversation.experiment_id is None:
            raise Refused("this conversation has no dataset yet; name a demo dataset, or ask "
                          "the person to supply theirs")
        exp = context.experiment(context.conversation.experiment_id, VIEW, BOTH)
        path, target, experiment_id = exp.data.dataset_path(), exp.data.target_column, exp.pk
        if path is None or not path.is_file():
            raise Refused("the experiment's dataset is not on this instance")
    facts = profile(path, target)
    return ToolResult(output=json.dumps(facts, default=str), experiment_id=experiment_id)


PROFILE_DATASET = ToolSpec(
    name="profile_dataset",
    description=(
        "Facts about the dataset this conversation is about: rows, each column's kind "
        "(numeric, categorical, datetime) as Codesigner reads it, missing values, "
        "distinct values, a few example values, the target column with the task it "
        "suggests, and the columns rows could be ordered in time by. Call it before "
        "saying anything about the data; never guess. With `demo`, a bundled demo "
        "dataset instead."),
    parameters={"type": "object",
                "properties": {"demo": {"type": "string",
                                        "description": "Name of a bundled demo dataset."}}},
    handler=_profile_dataset,
)


def _present_choices(context, question: str, options: list) -> ToolResult:
    return ToolResult(
        output="The choices are shown to the person. Call no more tools: their answer "
               "arrives as their next message.",
        waits_for_person=True,
        show={"question": question, "options": [str(o) for o in options]})


PRESENT_CHOICES = ToolSpec(
    name="present_choices",
    description=(
        "Ask the person to pick one of a few options, shown as buttons. Use it whenever "
        "a short list covers the answer. Then stop: their choice arrives as their next "
        "message."),
    parameters={"type": "object",
                "properties": {"question": {"type": "string"},
                               "options": {"type": "array", "items": {"type": "string"}}},
                "required": ["question", "options"]},
    handler=_present_choices,
)


#: How the person does what the assistant may not do for them. Plain steps, in
#: the words the pages use.
HOW = {
    "run": "On the experiment's page, fill in when the run should stop under "
           "\"Run until…\" in the sidebar, then press \"▶ Run\".",
    "cancel": "On the experiment's page, while it runs, press \"Cancel run\". It stops after "
              "the trial it is on.",
    "delete": "On the experiment's page, press \"🗑 Delete\" and confirm. It goes to the bin, "
              "from where it can be restored for a while.",
    "discard": "On any step of the draft's setup, press \"🗑 Discard draft\". A draft is "
               "thrown away for good.",
    "share": "On the experiment's page, open who it belongs to and share it with people in "
             "its group, as viewers or contributors. Only its owner can.",
    "transfer": "On the experiment's page, open who it belongs to and hand it to somebody "
                "else in its group. Only its owner can.",
    "groups": "Group leads add and remove people, and change roles, on the group's page "
              "under Groups.",
    "password": "Open the account page from the rail and press \"Change password\".",
    "account": "Account details are on the account page, from the rail; a site "
               "administrator changes what cannot be changed there.",
    "custom_model": "Upload the model's .py file in the first step of a new experiment's "
                    "setup, where the instance allows it.",
    "defaults": "Instance-wide default settings are changed on the default settings page, "
                "by somebody allowed to.",
}


def _explain_how(context, action: str) -> ToolResult:
    return ToolResult(output=HOW[action] + " Tell the person this; you cannot do it for them.")


EXPLAIN_HOW = ToolSpec(
    name="explain_how",
    description=(
        "How the person does something only they may do: start or cancel a run, delete "
        "an experiment or discard a draft, share or hand over an experiment, manage "
        "groups, change their password or account, upload model code, or change the "
        "instance's defaults. Use it instead of attempting any of these."),
    parameters={"type": "object",
                "properties": {"action": {"type": "string", "enum": sorted(HOW)}},
                "required": ["action"]},
    handler=_explain_how,
)


def basic():
    """The tools every conversation has."""
    return [PROFILE_DATASET, PRESENT_CHOICES, EXPLAIN_HOW]
