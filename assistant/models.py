"""`assistant.models` stores the assistant's chats, tool-call audit and
experiment notepads, which no `.ihpo` file contains.
"""

from django.conf import settings
from django.db import models


class Conversation(models.Model):
    """`Conversation` is one person's chat with the assistant."""

    PLANNING = "planning"
    SETUP = "setup"
    DASHBOARD = "dashboard"
    PHASES = [(PLANNING, "planning"), (SETUP, "setup"), (DASHBOARD, "dashboard")]

    #: `owner` is None on an instance without accounts.
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.CASCADE, related_name="conversations")
    #: With accounts on, `.Context.request` raises `.Refused` once the browser
    #: session `session_key` signs out. `session_key` is empty for a terminal
    #: chat.
    session_key = models.CharField(max_length=40, blank=True, default="")
    #: `browser` names the browser session the person last wrote from, and
    #: `.acts.send` addresses page actions to it.
    browser = models.CharField(max_length=40, blank=True, default="")
    experiment = models.ForeignKey("experiments.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="conversations")
    phase = models.CharField(max_length=20, choices=PHASES, default=PLANNING)
    plan = models.JSONField(default=dict, blank=True)
    #: `demo_dataset` names a bundled demo, or `dataset` stores the upload, until
    #: the plan is approved and a draft gets the dataset.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    dataset = models.FileField(upload_to="assistant/datasets/", blank=True, null=True)
    #: `dataset_filename` keeps the upload's original name.
    dataset_filename = models.CharField(max_length=255, blank=True, default="")
    #: `group` picks the experiment's group for a person in several.
    group = models.ForeignKey("access.Group", null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="+")
    #: `data_brief` caches the facts of `.brief_for` for one file and target.
    data_brief = models.JSONField(default=dict, blank=True)
    #: `turn_started_at` is when the running turn began; `.turns.busy` treats a
    #: turn older than `.turns.STALE` as over.
    turn_started_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def dataset_path(self):
        """`dataset_path` returns the planned dataset's `Path`, or None."""
        from pathlib import Path

        from core.datasets import demo_datasets

        if self.demo_dataset:
            path = demo_datasets().get(self.demo_dataset)
            return Path(path) if path else None
        return Path(self.dataset.path) if self.dataset else None

    @property
    def dataset_name(self) -> str:
        if self.demo_dataset:
            return self.demo_dataset
        if not self.dataset:
            return ""
        return self.dataset_filename or self.dataset.name.rsplit("/", 1)[-1]

    def forget_dataset(self) -> None:
        """`forget_dataset` clears the planned dataset and deletes its upload."""
        if self.dataset:
            self.dataset.delete(save=False)
        self.demo_dataset = self.dataset_filename = ""


class Message(models.Model):
    """`Message` is one message as the model receives it, from the person, the
    assistant or a tool."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ROLES = [(USER, "user"), (ASSISTANT, "assistant"), (TOOL, "tool")]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="messages")
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField(blank=True, default="")
    #: `tool_calls` uses the OpenAI wire format.
    tool_calls = models.JSONField(null=True, blank=True)
    tool_call_id = models.CharField(max_length=100, blank=True, default="")
    #: `tokens` counts an assistant message's model tokens for the daily budget.
    tokens = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def as_openai(self) -> dict:
        if self.role == self.TOOL:
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        message = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            message["content"] = self.content or None
        return message


class AssistantEvent(models.Model):
    """`AssistantEvent` is one item a chat page shows, such as text or a
    question. A reconnecting page asks for events after the last id it saw."""

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="events")
    kind = models.CharField(max_length=30)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class AgentAction(models.Model):
    """`AgentAction` audits one tool call and its outcome. A record outlives its
    conversation, because a chat reset leaves done actions in effect."""

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
    """`BrowserPage` is the page a browser session last reported open, with its
    fields, `data-assist` targets and figure tabs. The assistant never treats
    the record as permission."""

    browser = models.CharField(max_length=40, unique=True)
    path = models.CharField(max_length=500)
    url_name = models.CharField(max_length=100, blank=True, default="")
    experiment_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=300, blank=True, default="")
    fields = models.JSONField(default=list, blank=True)
    targets = models.JSONField(default=list, blank=True)
    #: `view` stores a figure page's `tab`, `tabs`, `outside_tabs`, `trials`
    #: (`{count, first, last}`) and `selected_trial`; other pages leave `view`
    #: empty.
    view = models.JSONField(default=dict, blank=True)
    #: Pages report on every change, so `reported_at` stays current.
    reported_at = models.DateTimeField(null=True, blank=True)
    seen_at = models.DateTimeField()
    #: `visitor` is the chat whose `navigate` took the browser off its own
    #: experiment's pages, and `return_path` is the page it left. Only the
    #: server sets them.
    visitor = models.ForeignKey(Conversation, null=True, blank=True,
                                on_delete=models.SET_NULL, related_name="+")
    return_path = models.CharField(max_length=500, blank=True, default="")


class ClientAct(models.Model):
    """`ClientAct` is one assistant action on the person's page plus the page's
    report. Any open page of the browser session carries the act out."""

    NAVIGATE = "navigate"
    FILL = "fill"
    HIGHLIGHT = "highlight"
    #: `REPORT` asks the page to describe itself.
    REPORT = "report"
    KINDS = [(NAVIGATE, "navigate"), (FILL, "fill"), (HIGHLIGHT, "highlight"), (REPORT, "report")]

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    STATUSES = [(PENDING, "pending"), (DONE, "done"), (FAILED, "failed")]

    browser = models.CharField(max_length=40, db_index=True)
    conversation = models.ForeignKey(Conversation, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="acts")
    kind = models.CharField(max_length=20, choices=KINDS)
    args = models.JSONField(default=dict, blank=True)
    #: `path` is the act's page, or empty for any page.
    path = models.CharField(max_length=500, blank=True, default="")
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class ExperimentNotes(models.Model):
    """`ExperimentNotes` is an experiment's Markdown notepad for later chats and
    collaborators, shown at ui:experiment_notes. Viewers read it; editors and
    their assistants write it."""

    #: `MAX_LENGTH` counts characters.
    MAX_LENGTH = 20000

    experiment = models.OneToOneField("experiments.Experiment", on_delete=models.CASCADE,
                                      related_name="notepad")
    text = models.TextField(blank=True, default="")
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="+")
    #: `by_assistant` is True when the last change came from `conversation`.
    by_assistant = models.BooleanField(default=False)
    conversation = models.ForeignKey(Conversation, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="+")
    updated_at = models.DateTimeField(auto_now=True)
