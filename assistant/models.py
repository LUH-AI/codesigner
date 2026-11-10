"""`assistant.models` stores the assistant's chats, tool-call audits and
notepads, which `.ihpo` files exclude.
"""

from django.conf import settings
from django.db import models


class Conversation(models.Model):

    PLANNING = "planning"
    SETUP = "setup"
    DASHBOARD = "dashboard"
    PHASES = [(PLANNING, "planning"), (SETUP, "setup"), (DASHBOARD, "dashboard")]

    #: `owner` is None without accounts.
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.CASCADE, related_name="conversations")
    #: `.Context.request` raises `.Refused` once the person signs out of `session_key`.
    #: Terminal chats leave `session_key` empty.
    session_key = models.CharField(max_length=40, blank=True, default="")
    #: `.acts.send` addresses page actions to the person's last browser session.
    browser = models.CharField(max_length=40, blank=True, default="")
    experiment = models.ForeignKey("experiments.Experiment", null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="conversations")
    phase = models.CharField(max_length=20, choices=PHASES, default=PLANNING)
    plan = models.JSONField(default=dict, blank=True)
    #: `demo_dataset` or `dataset` keeps the dataset until an approved plan's
    #: draft gets it.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    dataset = models.FileField(upload_to="assistant/datasets/", blank=True, null=True)
    dataset_filename = models.CharField(max_length=255, blank=True, default="")
    #: `group` picks the group for multi-group people.
    group = models.ForeignKey("access.Group", null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="+")
    #: Cached `.brief_for` facts for one file and target.
    data_brief = models.JSONField(default=dict, blank=True)
    #: `.turns.busy` treats a turn older than `.turns.STALE` as over.
    turn_started_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def dataset_path(self):
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
        """`forget_dataset` leaves saving to the caller."""
        if self.dataset:
            self.dataset.delete(save=False)
        self.demo_dataset = self.dataset_filename = ""


class Message(models.Model):
    """`Message` stores one chat message as the model receives it."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ROLES = [(USER, "user"), (ASSISTANT, "assistant"), (TOOL, "tool")]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="messages")
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField(blank=True, default="")
    #: OpenAI wire format.
    tool_calls = models.JSONField(null=True, blank=True)
    tool_call_id = models.CharField(max_length=100, blank=True, default="")
    #: Counted against the daily budget.
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
    """A reconnecting page asks for events after the last id it saw."""

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name="events")
    kind = models.CharField(max_length=30)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class AgentAction(models.Model):
    """Chat resets keep `AgentAction` rows, since done actions stay in effect."""

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
    """The assistant never treats `BrowserPage` as permission."""

    browser = models.CharField(max_length=40, unique=True)
    path = models.CharField(max_length=500)
    url_name = models.CharField(max_length=100, blank=True, default="")
    experiment_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=300, blank=True, default="")
    fields = models.JSONField(default=list, blank=True)
    targets = models.JSONField(default=list, blank=True)
    #: `view` stores a figure page's `tab`, `tabs`, `outside_tabs`, `trials`
    #: (`{count, first, last}`) and `selected_trial`.
    view = models.JSONField(default=dict, blank=True)
    #: `reported_at` is the time of the page's latest report.
    reported_at = models.DateTimeField(null=True, blank=True)
    seen_at = models.DateTimeField()
    #: `visitor` is the chat whose `navigate` took the browser elsewhere;
    #: `return_path` leads back. The server alone sets both.
    visitor = models.ForeignKey(Conversation, null=True, blank=True,
                                on_delete=models.SET_NULL, related_name="+")
    return_path = models.CharField(max_length=500, blank=True, default="")


class ClientAct(models.Model):
    """`ClientAct` is one assistant action on the person's page. The act runs
    only on the session's acting page, the one focused last. That page fails the
    act when `path` names another page."""

    NAVIGATE = "navigate"
    FILL = "fill"
    HIGHLIGHT = "highlight"
    #: The page describes itself.
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
    #: Empty `path` targets any page.
    path = models.CharField(max_length=500, blank=True, default="")
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class ExperimentNotes(models.Model):
    """`ExperimentNotes` is a Markdown notepad that editors and their assistants
    write."""

    #: In characters.
    MAX_LENGTH = 20000

    experiment = models.OneToOneField("experiments.Experiment", on_delete=models.CASCADE,
                                      related_name="notepad")
    text = models.TextField(blank=True, default="")
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="+")
    #: True when `conversation` made the last change.
    by_assistant = models.BooleanField(default=False)
    conversation = models.ForeignKey(Conversation, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="+")
    updated_at = models.DateTimeField(auto_now=True)
