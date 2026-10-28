"""`assistant.models` stores what the assistant keeps: each conversation,
every message in it as the model saw it, the events a page is shown, and a
record of every tool call.

The records stay on the serving instance and belong to no experiment, so no
`.ihpo` file contains them.
"""

from django.conf import settings
from django.db import models


class Conversation(models.Model):
    """`Conversation` is one person's conversation with the assistant, about one
    experiment once the experiment exists."""

    PLANNING = "planning"
    SETUP = "setup"
    DASHBOARD = "dashboard"
    PHASES = [(PLANNING, "planning"), (SETUP, "setup"), (DASHBOARD, "dashboard")]

    #: `owner` is None on an instance without accounts, which has one user.
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.CASCADE, related_name="conversations")
    #: `session_key` names the browser session the conversation was started
    #: from. The conversation's tools act only while that session is signed in
    #: as the owner, so signing out or an expired session stops the assistant
    #: acting for anybody. `session_key` is empty for a conversation started
    #: from the terminal by somebody who already has the server.
    session_key = models.CharField(max_length=40, blank=True, default="")
    #: `browser` names the browser session the person last wrote from, and the
    #: assistant's actions on a page go there (assistant/acts.py). `browser` is
    #: kept with or without accounts.
    browser = models.CharField(max_length=40, blank=True, default="")
    experiment = models.ForeignKey("experiments.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="conversations")
    phase = models.CharField(max_length=20, choices=PHASES, default=PLANNING)
    #: `plan` holds the plan agreed with the person; the assistant fills the
    #: forms from it.
    plan = models.JSONField(default=dict, blank=True)
    #: `demo_dataset` names a bundled demo, or `dataset` holds the person's
    #: upload, while the plan is being made. A draft gets made, with the dataset
    #: on it, only once the plan is approved, as setup step 1 would.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    dataset = models.FileField(upload_to="assistant/datasets/", blank=True, null=True)
    #: `dataset_filename` keeps the upload's original name, since storage may
    #: store the file under another.
    dataset_filename = models.CharField(max_length=255, blank=True, default="")
    #: `group` is the group the experiment will go in, for a person with more
    #: than one.
    group = models.ForeignKey("access.Group", null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="+")
    #: `data_brief` caches what the dataset says while the file stays the same
    #: (`assistant.tools.data.brief_for`): {"for": file and target, "facts"}.
    data_brief = models.JSONField(default=dict, blank=True)
    #: `turn_started_at` is when the running turn began, or None. Only one turn
    #: runs at a time, and a turn left by a stopped server counts as over
    #: (assistant/turns.py).
    turn_started_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def dataset_path(self):
        """`dataset_path` returns the planned dataset's file as a `Path`, or
        None without one."""
        from pathlib import Path

        from core.datasets import demo_datasets

        if self.demo_dataset:
            path = demo_datasets().get(self.demo_dataset)
            return Path(path) if path else None
        return Path(self.dataset.path) if self.dataset else None

    @property
    def dataset_name(self) -> str:
        """`dataset_name` is the name the chat uses for the planned dataset."""
        if self.demo_dataset:
            return self.demo_dataset
        if not self.dataset:
            return ""
        return self.dataset_filename or self.dataset.name.rsplit("/", 1)[-1]

    def forget_dataset(self) -> None:
        """`forget_dataset` drops the planned dataset, and its file if it is an
        upload."""
        if self.dataset:
            self.dataset.delete(save=False)
        self.demo_dataset = self.dataset_filename = ""


class Message(models.Model):
    """`Message` is one message of a conversation as the model is sent it: the
    person's, the assistant's (with any tool calls it made), and each tool's
    answer."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ROLES = [(USER, "user"), (ASSISTANT, "assistant"), (TOOL, "tool")]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="messages")
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField(blank=True, default="")
    #: The calls an assistant message made, in the OpenAI wire format.
    tool_calls = models.JSONField(null=True, blank=True)
    #: The call a tool message answers.
    tool_call_id = models.CharField(max_length=100, blank=True, default="")
    #: `tokens` counts the model tokens an assistant message cost; the daily
    #: budget is counted in them.
    tokens = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def as_openai(self) -> dict:
        """`as_openai` returns the message in the OpenAI chat format."""
        if self.role == self.TOOL:
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        message = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            message["content"] = self.content or None
        return message


class AssistantEvent(models.Model):
    """`AssistantEvent` is something a page showing the conversation should
    show: text, a tool at work, a question with choices, the end of a turn.
    Event ids order the events, so a page that reconnects asks for the ones
    after the last it saw."""

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="events")
    kind = models.CharField(max_length=30)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class AgentAction(models.Model):
    """`AgentAction` records one tool call, who it was for, and what came of
    it; the records audit everything the assistant did or tried. A record
    survives deletion of its conversation, because a chat can be reset and
    what was done in it cannot."""

    OK = "ok"
    REFUSED = "refused"
    INVALID = "invalid"
    FAILED = "failed"
    OUTCOMES = [(OK, "ok"), (REFUSED, "refused"), (INVALID, "invalid"), (FAILED, "failed")]

    conversation = models.ForeignKey(Conversation, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="actions")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.SET_NULL, related_name="+")
    tool = models.CharField(max_length=100)
    arguments = models.JSONField(default=dict, blank=True)
    outcome = models.CharField(max_length=10, choices=OUTCOMES)
    detail = models.TextField(blank=True, default="")
    experiment = models.ForeignKey("experiments.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="+")
    duration_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class BrowserPage(models.Model):
    """`BrowserPage` is the Codesigner page a browser session last had open, as
    the page reported it: which page, its form's fields and values, and the
    elements it names for the assistant (`data-assist`). The assistant reads
    the record and never trusts it as permission for anything."""

    browser = models.CharField(max_length=40, unique=True)
    path = models.CharField(max_length=500)
    #: `url_name` is the URL's name, e.g. "ui:setup_optimizer", or empty for an
    #: unnamed path.
    url_name = models.CharField(max_length=100, blank=True, default="")
    experiment_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=300, blank=True, default="")
    fields = models.JSONField(default=list, blank=True)
    targets = models.JSONField(default=list, blank=True)
    #: When a page of the session last showed it was open.
    seen_at = models.DateTimeField()


class ClientAct(models.Model):
    """`ClientAct` is something the assistant does on the person's page (go to
    a page, fill fields, point something out) and what the page reported back.
    An act is addressed to a browser session, so whichever of its pages is open
    carries it out."""

    NAVIGATE = "navigate"
    FILL = "fill"
    HIGHLIGHT = "highlight"
    KINDS = [(NAVIGATE, "navigate"), (FILL, "fill"), (HIGHLIGHT, "highlight")]

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    STATUSES = [(PENDING, "pending"), (DONE, "done"), (FAILED, "failed")]

    browser = models.CharField(max_length=40, db_index=True)
    conversation = models.ForeignKey(Conversation, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="acts")
    kind = models.CharField(max_length=20, choices=KINDS)
    args = models.JSONField(default=dict, blank=True)
    #: `path` is the page the act is meant for, or empty for an act that goes
    #: anywhere.
    path = models.CharField(max_length=500, blank=True, default="")
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
