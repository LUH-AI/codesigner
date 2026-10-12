"""What the assistant keeps: each conversation, every message in it as the
model saw it, the events a page is shown, and a record of every tool call.

All of it stays on this instance. None of it is part of an experiment, so none
of it is in an `.ihpo`.
"""

from django.conf import settings
from django.db import models


class Conversation(models.Model):
    """One person's conversation with the assistant, about one experiment once
    there is one."""

    PLANNING = "planning"
    SETUP = "setup"
    DASHBOARD = "dashboard"
    PHASES = [(PLANNING, "planning"), (SETUP, "setup"), (DASHBOARD, "dashboard")]

    #: None on an instance without accounts, where there is one somebody.
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.CASCADE, related_name="conversations")
    #: The browser session the conversation was started from. Its tools act only
    #: while that session is signed in as the owner: signing out, or the session
    #: expiring, stops an assistant acting for anybody. Empty for one started
    #: from the terminal, by somebody who already has the server.
    session_key = models.CharField(max_length=40, blank=True, default="")
    #: The browser session the person last wrote from: where the assistant's
    #: actions on a page go (assistant/acts.py). Kept with or without accounts.
    browser = models.CharField(max_length=40, blank=True, default="")
    experiment = models.ForeignKey("ui.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="conversations")
    phase = models.CharField(max_length=20, choices=PHASES, default=PLANNING)
    #: The plan agreed with the person; what the assistant fills the forms from.
    plan = models.JSONField(default=dict, blank=True)
    #: The dataset the plan is for, while it is being planned: a bundled demo by
    #: name, or the file the person uploaded. Only when the plan is approved is
    #: a draft made and the dataset put on it, as setup step 1 would.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    dataset = models.FileField(upload_to="assistant/datasets/", blank=True, null=True)
    #: The upload's name as the person's file had it; storage may have to
    #: store it under another.
    dataset_filename = models.CharField(max_length=255, blank=True, default="")
    #: The group the experiment will go in, for a person with more than one.
    group = models.ForeignKey("access.Group", null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="+")
    #: When the turn now running began, or None: one turn at a time, and one
    #: left by a server that stopped is taken as over (assistant/turns.py).
    turn_started_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def dataset_path(self):
        """The planned dataset's file, as a `Path`, or None without one."""
        from pathlib import Path

        from core.io import demo_datasets

        if self.demo_dataset:
            path = demo_datasets().get(self.demo_dataset)
            return Path(path) if path else None
        return Path(self.dataset.path) if self.dataset else None

    @property
    def dataset_name(self) -> str:
        """What the chat calls the planned dataset."""
        if self.demo_dataset:
            return self.demo_dataset
        if not self.dataset:
            return ""
        return self.dataset_filename or self.dataset.name.rsplit("/", 1)[-1]

    def forget_dataset(self) -> None:
        """Drop the planned dataset, and its file if it is an upload."""
        if self.dataset:
            self.dataset.delete(save=False)
        self.demo_dataset = self.dataset_filename = ""


class Message(models.Model):
    """One message of a conversation, exactly as the model is sent it: the
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
    #: Model tokens this message cost, for an assistant message: what the
    #: daily budget is counted in.
    tokens = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def as_openai(self) -> dict:
        """The message in the OpenAI chat format."""
        if self.role == self.TOOL:
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        message = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            message["content"] = self.content or None
        return message


class AssistantEvent(models.Model):
    """Something a page showing the conversation should show: text, a tool at
    work, a question with choices, the end of a turn. Its id orders them, so a
    page that reconnects asks for the ones after the last it saw."""

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="events")
    kind = models.CharField(max_length=30)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class AgentAction(models.Model):
    """One tool call, who it was for, and what came of it: the audit of
    everything the assistant did or tried. Kept when its conversation is
    deleted: a chat can be reset, what was done in it cannot."""

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
    experiment = models.ForeignKey("ui.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="+")
    duration_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class BrowserPage(models.Model):
    """The Codesigner page a browser session last had in front of it, as the
    page itself reported it: which page, its form's fields and values, and the
    elements it names for the assistant (`data-assist`). What the assistant
    looks at; never trusted as permission for anything."""

    browser = models.CharField(max_length=40, unique=True)
    path = models.CharField(max_length=500)
    #: The URL's name, e.g. "ui:setup_optimizer"; empty for a path that is none.
    url_name = models.CharField(max_length=100, blank=True, default="")
    experiment_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=300, blank=True, default="")
    fields = models.JSONField(default=list, blank=True)
    targets = models.JSONField(default=list, blank=True)
    #: When a page of the session last showed it was open.
    seen_at = models.DateTimeField()


class ClientAct(models.Model):
    """Something the assistant does on the person's page — go to a page, fill
    fields, point something out — and what the page reported back. Addressed
    to a browser session, so whichever of its pages is open carries it out."""

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
    #: The page it is meant for, or empty for one that goes anywhere.
    path = models.CharField(max_length=500, blank=True, default="")
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
