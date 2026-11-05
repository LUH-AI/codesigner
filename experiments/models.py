from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from django.conf import settings as django_settings
from django.db import models, transaction
from django.utils.translation import gettext_lazy as _

from core import tasks

from .fields import SafeJSONField

if TYPE_CHECKING:
    from django.db.models.manager import RelatedManager

    from ui.models import FigureLayout


UNTITLED = _("Untitled Experiment")


def account(user):
    return user if getattr(user, "is_authenticated", False) else None


def generate_identifier() -> str:
    """`generate_identifier` returns eight hex characters.
    """
    return secrets.token_hex(4)


class ExperimentData(models.Model):
    """`ExperimentData` stores the portable half, saved in every `.ihpo`.
    """

    id: int
    experiment: Experiment
    history: RelatedManager[ExperimentEvent]

    name = models.CharField(max_length=200, blank=True, default="")
    #: Exported setup time; `Experiment.created_at` records local arrival.
    created_at = models.DateTimeField(null=True, blank=True)
    #: An imported file sets `time_basis`, the form every page shows times in.
    time_basis = models.CharField(
        max_length=10, default="absolute",
        choices=[("absolute", "absolute"), ("relative", "relative"), ("none", "none")])
    model_name = models.CharField(max_length=200)
    model_file = models.FileField(upload_to="custom_models/", blank=True, null=True)
    optimizer_name = models.CharField(max_length=200)
    optimizer_params = models.JSONField(default=dict, blank=True)
    metric_names = models.JSONField(default=list)
    #: `current_metric` is the optimizer's target, and `original_metric` pins
    #: the first run's.
    current_metric = models.CharField(max_length=100, blank=True, null=True)
    original_metric = models.CharField(max_length=100, blank=True, null=True)
    seed = models.IntegerField(default=0)
    #: Zero gives a holdout and k >= 2 gives k-fold CV, fixed for comparable
    #: trials.
    cv_folds = models.IntegerField(default=0)
    #: Unused under cross-validation.
    test_size = models.FloatField(default=0.2)
    #: `time_gap` rows separate training from validation in a time split.
    time_column = models.CharField(max_length=255, blank=True, default="")
    time_gap = models.PositiveIntegerField(default=0)
    #: A nonzero `horizon` makes a forecast scored by backtests.
    horizon = models.PositiveIntegerField(default=0)
    series_column = models.CharField(max_length=255, blank=True, default="")
    season = models.PositiveIntegerField(default=0)
    #: Fixed for the experiment's life.
    task = models.CharField(max_length=20, default=tasks.DEFAULT,
                            choices=[(t, t) for t in tasks.TASKS])
    dataset = models.FileField(upload_to="datasets/", blank=True, null=True)
    #: Instances resolve demo names to their own copies.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    #: An empty `target_column` predicts the dataset's last column.
    target_column = models.CharField(max_length=255, blank=True, default="")
    #: Empty `processing` applies each step as the model needs
    #: (`core.processing`).
    processing = models.JSONField(default=dict, blank=True)
    #: A run saves `config_space` because a read-only page cannot ask a custom
    #: model.
    config_space: models.JSONField[dict | None] = models.JSONField(blank=True, null=True, default=None)
    result = SafeJSONField(blank=True, null=True, default=None)
    #: `priors` maps each hyperparameter to its stated prior. `.SafeJSONField`
    #: stores the infinity a spike can produce.
    priors = SafeJSONField(default=dict, blank=True)
    #: `trial_budget` counts trials planned across all runs; null means no
    #: budget.
    trial_budget = models.PositiveIntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return self.name or str(UNTITLED)

    def dataset_path(self):
        """`dataset_path` prefers the demo, then the upload."""
        from pathlib import Path

        from core.datasets import demo_datasets

        if self.demo_dataset:
            path = demo_datasets().get(self.demo_dataset)
            return Path(path) if path else None
        if self.dataset:
            return Path(self.dataset.path)
        return None

    @property
    def has_dataset(self) -> bool:
        return bool(self.demo_dataset or self.dataset)

    @property
    def forecasts(self) -> bool:
        return self.horizon > 0


class ExperimentManager(models.Manager):
    """`ExperimentManager.create` splits fields between both halves and writes
    them in one transaction.
    """

    def create(self, **fields):
        mine = {f.name for f in ExperimentData._meta.get_fields()
                if getattr(f, "concrete", False)}
        theirs = {key: fields.pop(key) for key in list(fields) if key in mine}
        with transaction.atomic():
            return super().create(data=ExperimentData.objects.create(**theirs),
                                  **fields)


class Experiment(models.Model):
    """`Experiment` keeps instance-only state, and `data` keeps the portable
    half.
    """

    id: int
    data_id: int
    owner_id: int | None
    group_id: int | None
    deleted_by_id: int | None
    shares: RelatedManager[ExperimentShare]
    bin_entries: RelatedManager[BinEntry]
    runs: RelatedManager[Run]
    trial_models: RelatedManager[TrialModel]
    figure_layouts: RelatedManager[FigureLayout]

    #: Deleting the data deletes the experiment.
    data = models.OneToOneField(ExperimentData, on_delete=models.CASCADE,
                                related_name="experiment")
    identifier = models.CharField(max_length=12, unique=True, editable=False, default=generate_identifier)
    created_at = models.DateTimeField(auto_now_add=True)
    #: Settings stay local; an imported `.ihpo` gets local defaults.
    settings = models.JSONField(default=dict, blank=True)
    use_default_settings = models.BooleanField(default=True)

    #: Null means unowned; SET_NULL keeps results.
    owner = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="experiments")
    #: `group` is stored because memberships change or vanish. `.Group.delete`
    #: bins the work first, since PROTECT blocks the deletion. A null group with
    #: `trashed_at` set means a deleted group.
    group = models.ForeignKey(
        "access.Group", null=True, blank=True,
        on_delete=models.PROTECT, related_name="experiments")
    #: `group_share` shares the experiment with everyone in its group, people
    #: who join later included, at an `ExperimentShare` level; empty shares it
    #: with nobody. A person's own share counts when it is the higher level.
    group_share = models.CharField(
        max_length=16, blank=True, default="",
        choices=[("viewer", _("Viewer")), ("contributor", _("Contributor"))])
    #: Group deletion sets `trashed_at`, which hides the experiment until
    #: rehoming clears it.
    trashed_at = models.DateTimeField(null=True, blank=True)
    #: Personal deletion (`BinEntry`) keeps all data, so restoring clears
    #: `deleted_at` and `deleted_by`.
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="+")
    #: A draft is only reachable through its setup (ui/setup_steps.py).
    draft = models.BooleanField(default=False)
    setup_saved = models.JSONField(default=dict, blank=True)
    # Statuses from services/modelenv.py.
    ENV_NONE = "none"
    ENV_PENDING = "pending"
    ENV_PREPARING = "preparing"
    ENV_READY = "ready"
    ENV_FAILED = "failed"
    ENV_SKIPPED = "skipped"
    ENV_LEGACY = "legacy"
    ENV_STATUS_CHOICES = [
        (ENV_NONE, "Not required"), (ENV_PENDING, "Queued"),
        (ENV_PREPARING, "Preparing"), (ENV_READY, "Ready"),
        (ENV_FAILED, "Failed"), (ENV_SKIPPED, "No uv — runs in-process"),
        (ENV_LEGACY, "Predates environments"),
    ]
    env_status = models.CharField(max_length=20, choices=ENV_STATUS_CHOICES, default=ENV_NONE)
    env_error = models.TextField(blank=True, default="")
    #: Dependencies, interpreter, lock digest.
    env_meta = models.JSONField(default=dict, blank=True)
    env_prepared_at = models.DateTimeField(null=True, blank=True)

    objects = ExperimentManager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    def refresh_from_db(self, *args, **kwargs):
        """`refresh_from_db` drops the cached `data`, so re-reads see the latest
        `result`.
        """
        vars(self).pop("data", None)
        return super().refresh_from_db(*args, **kwargs)

    @property
    def name(self) -> str:
        """`name` reads `data.name` so no copy can disagree; queries use
        `data__name`.
        """
        return self.data.name

    @property
    def title(self) -> str:
        """`title` falls back to `UNTITLED`."""
        return self.data.name or str(UNTITLED)

    def save(self, *args, **kwargs):
        """`save` only assigns a new experiment's group when its owner has a
        single group, since several paths create experiments.
        """
        if self._state.adding and self.group_id is None and self.owner_id:
            assert self.owner is not None
            groups = list(self.owner.memberships.values_list("group_id", flat=True)[:2])
            if len(groups) == 1:
                self.group_id = groups[0]
        return super().save(*args, **kwargs)

    @property
    def is_running(self) -> bool:
        return self.runs.filter(status__in=Run.ACTIVE_STATUSES).exists()

    @property
    def env_pending(self) -> bool:
        return self.env_status in (self.ENV_PENDING, self.ENV_PREPARING)

    @property
    def env_in_process(self) -> bool:
        return self.env_status in (self.ENV_SKIPPED, self.ENV_LEGACY)


class ExperimentShare(models.Model):
    """Grants are per person because memberships change; leaving the group drops
    the grants (`access/signals.py`).
    """

    id: int
    experiment_id: int
    user_id: int
    granted_by_id: int | None

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    LEVELS = [
        # Viewers see and export.
        (VIEWER, _("Viewer")),
        # Contributors also run, configure and delete.
        (CONTRIBUTOR, _("Contributor")),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,
                                   related_name="shares")
    user = models.ForeignKey(django_settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE,
                             related_name="experiment_shares")
    level = models.CharField(max_length=16, choices=LEVELS, default=VIEWER)
    granted_at = models.DateTimeField(auto_now_add=True)
    #: The owner at grant time.
    granted_by = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True,
                                   blank=True, on_delete=models.SET_NULL,
                                   related_name="+")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["experiment", "user"],
                                    name="one_share_per_person_per_experiment"),
        ]

    def __str__(self):
        return f"{self.user} → {self.experiment_id} ({self.level})"

    @property
    def is_contributor(self):
        return self.level == self.CONTRIBUTOR

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.experiment_id is None or self.user_id is None:
            return
        experiment = self.experiment
        if self.user_id == experiment.owner_id:
            raise ValidationError(_("The owner already has every kind of access."))
        if experiment.group_id is None or not self.user.memberships.filter(
                group_id=experiment.group_id).exists():
            from access.models import shown_name

            raise ValidationError(
                _("%(user)s is not in this experiment's group, so it cannot be "
                  "shared with them.") % {"user": shown_name(self.user)})


class ExperimentEvent(models.Model):
    """`ExperimentEvent` records one search-changing action; `payload` keeps
    whole before and after values.
    """

    id: int
    data_id: int
    run_id: int | None
    actor_id: int | None

    data = models.ForeignKey(ExperimentData, on_delete=models.CASCADE,
                             related_name="history")
    at = models.DateTimeField(null=True, blank=True)
    at_trial = models.IntegerField(default=0)
    kind = models.CharField(max_length=40)
    payload = models.JSONField(default=dict, blank=True)
    run = models.ForeignKey("Run", null=True, blank=True, on_delete=models.SET_NULL,
                            related_name="history")
    #: Instance-only.
    actor = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="experiment_events")

    class Meta:
        ordering = ["at", "id"]


class BinEntry(models.Model):
    """`BinEntry` records owners and contributors at deletion, so later grant
    changes leave bins alone.
    """

    id: int
    experiment_id: int
    user_id: int

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,
                                   related_name="bin_entries")
    user = models.ForeignKey(django_settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE, related_name="bin_entries")
    binned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["experiment", "user"],
                                    name="one_bin_entry_per_person_per_experiment"),
        ]


class Run(models.Model):
    """A background worker drives each `Run`, and pages observe or cancel it.
    """

    id: int
    experiment_id: int
    started_by_id: int | None
    cancel_requested_by_id: int | None
    history: RelatedManager[ExperimentEvent]
    trial_models: RelatedManager[TrialModel]

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("submitted", "Submitted"),
        ("running", "Running"),
        ("done", "Done"),
        ("error", "Error"),
        ("cancelled", "Cancelled"),
    ]
    ACTIVE_STATUSES = ("pending", "submitted", "running")

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="runs")
    #: The worker checks `started_by` for the custom-model permission, since
    #: unowned experiments lack owners.
    started_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="runs_started")
    stopping = models.JSONField(default=dict)
    #: `trial_timeout` is `{"mode", "seconds", "factor"}`, limits each model call
    #: and only ends the current trial; an empty dict means the default
    #: (`.as_deadline`).
    trial_timeout = models.JSONField(default=dict, blank=True)
    #: `priors` copies the experiment's priors at creation, since
    #: `ExperimentData.priors` changes with every drag.
    priors = models.JSONField(default=dict, blank=True)
    #: A copy at creation, since `.RUN_SAFE` settings change between runs.
    optimizer_params = models.JSONField(default=dict, blank=True)
    #: Page-picked configurations, each `{"config", "origin", "proposal"}`, run
    #: first.
    configurations = models.JSONField(default=list, blank=True)
    #: Ending criterion, else empty.
    stopped_by = models.CharField(max_length=40, blank=True, default="")
    primary_metric = models.CharField(max_length=100)
    #: With `trial_count`, gives this run's trial range.
    trial_offset = models.IntegerField(null=True, blank=True)
    #: Append-only search-level events.
    events = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    #: `backend` records where each run ran, since the configured backend can
    #: change later.
    backend = models.CharField(max_length=20, default="local")
    job_id = models.CharField(max_length=64, blank=True, default="")
    cancel_requested = models.BooleanField(default=False)
    #: `finished_at` is set once the trial in flight returns.
    cancel_requested_at = models.DateTimeField(null=True, blank=True)
    cancel_requested_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="runs_cancelled")
    error = models.TextField(blank=True, default="")
    #: Run press time; `started_at` may lag.
    created_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    #: Trial seconds; the remainder is overhead.
    trial_seconds = models.FloatField(null=True, blank=True)
    #: Null until finished.
    trial_count = models.IntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.experiment.title} run #{self.pk} ({self.status})"

    @property
    def max_trials(self):
        """`max_trials` gives the trial cap, else None."""
        return self.stopping.get("max_trials")

    @property
    def duration(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class TrialModel(models.Model):
    """`TrialModel` stores one row per kept model, so storage sums rows instead
    of walking the disk.
    """

    id: int
    experiment_id: int
    run_id: int | None

    experiment = models.ForeignKey("Experiment", on_delete=models.CASCADE,
                                   related_name="trial_models")
    run = models.ForeignKey("Run", on_delete=models.SET_NULL, null=True, blank=True,
                            related_name="trial_models")
    trial = models.PositiveIntegerField()
    directory = models.CharField(max_length=500)
    bytes = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["experiment", "trial"],
                                               name="one_kept_model_per_trial")]

    def __str__(self) -> str:
        return f"{self.experiment_id} trial {self.trial}"


class GlobalSettings(models.Model):
    """`GlobalSettings` keeps one row."""

    id: int

    default_experiment_settings = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name_plural = "Global settings"

    def __str__(self) -> str:
        return "Global settings"

    @classmethod
    def get_solo(cls) -> "GlobalSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
