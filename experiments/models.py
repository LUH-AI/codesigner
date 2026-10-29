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


#: Display name for an unnamed experiment.
UNTITLED = _("Untitled Experiment")


def account(user):
    """`account` returns the signed-in user, else None."""
    return user if getattr(user, "is_authenticated", False) else None


def generate_identifier() -> str:
    """`generate_identifier` makes eight hex characters, independent of the
    mutable name.
    """
    return secrets.token_hex(4)


class ExperimentData(models.Model):
    """`ExperimentData` stores the portable half: every field appears in the
    snapshot, and an `.ihpo` opened elsewhere rebuilds the row.

    `Experiment` stores the instance half, and
    `tests/ui/experiments/test_the_two_halves.py` fails on a field that belongs
    to neither.
    """

    id: int
    experiment: Experiment
    history: RelatedManager[ExperimentEvent]

    #: Optional; `Experiment.title` names an unnamed one.
    name = models.CharField(max_length=200, blank=True, default="")
    #: The setup time travels in the file, while `Experiment.created_at` records
    #: local arrival.
    created_at = models.DateTimeField(null=True, blank=True)
    #: `time_basis` is "absolute", "relative" or "none", set by an imported file,
    #: and pages show every time in the same form
    #: (experiments/services/timestamps.py).
    time_basis = models.CharField(
        max_length=10, default="absolute",
        choices=[("absolute", "absolute"), ("relative", "relative"), ("none", "none")])
    model_name = models.CharField(max_length=200)
    model_file = models.FileField(upload_to="custom_models/", blank=True, null=True)
    optimizer_name = models.CharField(max_length=200)
    optimizer_params = models.JSONField(default=dict, blank=True)
    metric_names = models.JSONField(default=list)
    #: `current_metric` is what the optimizer targets now, and `original_metric`
    #: is the first run's, pinned once. Per-run changes live in
    #: `Run.primary_metric` and `Run.events`.
    current_metric = models.CharField(max_length=100, blank=True, null=True)
    original_metric = models.CharField(max_length=100, blank=True, null=True)
    seed = models.IntegerField(default=0)
    #: A value of 0 is a single holdout and k >= 2 is k-fold CV, fixed at
    #: creation so trials stay comparable (core.splits).
    cv_folds = models.IntegerField(default=0)
    #: Holdout share, carried unused under cross-validation.
    test_size = models.FloatField(default=0.2)
    #: `time_column` orders a time split ("" ignores time), and `time_gap` counts
    #: the rows left between training and validation.
    time_column = models.CharField(max_length=255, blank=True, default="")
    time_gap = models.PositiveIntegerField(default=0)
    #: A nonzero `horizon` makes a forecast evaluated by backtests, and `season`
    #: stays 0 until inferred (core.forecasting).
    horizon = models.PositiveIntegerField(default=0)
    series_column = models.CharField(max_length=255, blank=True, default="")
    season = models.PositiveIntegerField(default=0)
    #: Class or number (core.tasks), fixed for the experiment's life.
    task = models.CharField(max_length=20, default=tasks.DEFAULT,
                            choices=[(t, t) for t in tasks.TASKS])
    dataset = models.FileField(upload_to="datasets/", blank=True, null=True)
    #: A bundled demo is shared by name, and another instance resolves the name
    #: to its own copy.
    demo_dataset = models.CharField(max_length=200, blank=True, default="")
    #: An empty `target_column` predicts the dataset's last column, as every
    #: experiment made before the choice existed does.
    target_column = models.CharField(max_length=255, blank=True, default="")
    #: The processing steps (core.processing) are part of the file, and empty
    #: means each step as the model needs.
    processing = models.JSONField(default=dict, blank=True)
    #: `config_space` stores ConfigSpace's serialized dict, null until supplied.
    #:
    #: A run saves the space (services/run.py) because a read-only page cannot
    #: ask a custom model, and an external import takes the space from its file.
    config_space: models.JSONField[dict | None] = models.JSONField(blank=True, null=True, default=None)
    result = SafeJSONField(blank=True, null=True, default=None)
    #: `priors` maps each hyperparameter to `{kind, params, exponent, decay,
    #: knots}`, stated on the acquisition figure for the next run.
    #:
    #: `knots` is the grid density computed once in the browser and sent to SMAC,
    #: so page and search never drift, and kind and params only restore the
    #: controls. SafeJSONField stores the infinity a spike can produce.
    priors = SafeJSONField(default=dict, blank=True)

    def __str__(self) -> str:
        return self.name or str(UNTITLED)

    def dataset_path(self):
        """`dataset_path` prefers the demo path, then the stored copy, else
        None."""
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
    """`ExperimentManager.create` routes flat fields to the half that declares
    them and writes both in one transaction, since `Experiment.data` requires a
    row.
    """

    def create(self, **fields):
        mine = {f.name for f in ExperimentData._meta.get_fields()
                if getattr(f, "concrete", False)}
        theirs = {key: fields.pop(key) for key in list(fields) if key in mine}
        with transaction.atomic():
            return super().create(data=ExperimentData.objects.create(**theirs),
                                  **fields)


class Experiment(models.Model):
    """`Experiment` stores instance-only state: owner, group, settings and the
    model environment. The work lives in `data`, and fields on `Experiment`
    concern the local instance only.
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

    #: `Experiment` and its `data` describe one experiment, so deleting the data
    #: deletes the row.
    data = models.OneToOneField(ExperimentData, on_delete=models.CASCADE,
                                related_name="experiment")
    identifier = models.CharField(max_length=12, unique=True, editable=False, default=generate_identifier)
    created_at = models.DateTimeField(auto_now_add=True)
    #: The overrides take effect unless `use_default_settings` is set. The
    #: settings stay local, so an imported `.ihpo` gets the receiving instance's
    #: defaults.
    settings = models.JSONField(default=dict, blank=True)
    use_default_settings = models.BooleanField(default=True)

    #: Null `owner` means unowned, and SET_NULL keeps results when a person is
    #: removed (access/policy.py).
    owner = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="experiments")
    #: `group` is stored because the owner's membership stops answering when an
    #: account is deleted or someone changes group. PROTECT blocks a group
    #: deletion, and `Group.delete()` bins the work first.
    #:
    #: A null group with `trashed_at` set means a deleted group, and without it
    #: an install with no accounts.
    group = models.ForeignKey(
        "access.Group", null=True, blank=True,
        on_delete=models.PROTECT, related_name="experiments")
    #: Group deletion sets `trashed_at`, and rehoming clears it; the policy hides
    #: the experiment until a site admin decides.
    trashed_at = models.DateTimeField(null=True, blank=True)
    #: A person's deletion, distinct from `trashed_at`, is settled from the
    #: people's own bins (`BinEntry`). Rows, runs, grants and files stay, so
    #: restoring clears `deleted_at` and `deleted_by`.
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="+")
    #: A draft is reachable only through its setup (ui/setup_steps.py), and
    #: `setup_saved` records each saved step's time.
    draft = models.BooleanField(default=False)
    setup_saved = models.JSONField(default=dict, blank=True)
    # Environment status (services/modelenv.py), locked once per experiment.
    ENV_NONE = "none"            # registry model
    ENV_PENDING = "pending"
    ENV_PREPARING = "preparing"
    ENV_READY = "ready"
    ENV_FAILED = "failed"
    ENV_SKIPPED = "skipped"      # no uv; runs in-process
    ENV_LEGACY = "legacy"        # predates environments
    ENV_STATUS_CHOICES = [
        (ENV_NONE, "Not required"), (ENV_PENDING, "Queued"),
        (ENV_PREPARING, "Preparing"), (ENV_READY, "Ready"),
        (ENV_FAILED, "Failed"), (ENV_SKIPPED, "No uv — runs in-process"),
        (ENV_LEGACY, "Predates environments"),
    ]
    env_status = models.CharField(max_length=20, choices=ENV_STATUS_CHOICES, default=ENV_NONE)
    env_error = models.TextField(blank=True, default="")
    #: Dependencies, interpreter, model class and lock digest.
    env_meta = models.JSONField(default=dict, blank=True)
    env_prepared_at = models.DateTimeField(null=True, blank=True)

    objects = ExperimentManager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    def refresh_from_db(self, *args, **kwargs):
        """`refresh_from_db` drops the cached `data` relation, so a re-read sees
        what a run wrote to `result`.
        """
        vars(self).pop("data", None)
        return super().refresh_from_db(*args, **kwargs)

    @property
    def name(self) -> str:
        """`name` reads `data.name` instead of copying it, so no second copy can
        disagree. Filters and ordering still use `data__name`.
        """
        return self.data.name

    @property
    def title(self) -> str:
        """`title` falls back to "Untitled Experiment" for an empty name."""
        return self.data.name or str(UNTITLED)

    def save(self, *args, **kwargs):
        """`save` puts a new experiment into its owner's group only when the
        owner belongs to one group, and the caller chooses otherwise. The rule
        lives on the model because several paths create experiments, and it
        only applies on creation.
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
        """`env_in_process` covers missing uv and pre-environment experiments
        alike.
        """
        return self.env_status in (self.ENV_SKIPPED, self.ENV_LEGACY)


class ExperimentShare(models.Model):
    """Grants are per person, since a group's membership changes. Only members
    of the experiment's group receive one, and leaving the group drops its grants
    (`access/signals.py`).
    """

    id: int
    experiment_id: int
    user_id: int
    granted_by_id: int | None

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    LEVELS = [
        # Viewers see the page and export.
        (VIEWER, _("Viewer")),
        # Contributors also run, configure and delete; sharing stays with the
        # owner.
        (CONTRIBUTOR, _("Contributor")),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,
                                   related_name="shares")
    user = models.ForeignKey(django_settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE,
                             related_name="experiment_shares")
    level = models.CharField(max_length=16, choices=LEVELS, default=VIEWER)
    granted_at = models.DateTimeField(auto_now_add=True)
    #: The owner at the time; grants survive ownership changes.
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
            return  # field errors cover it
        experiment = self.experiment
        if self.user_id == experiment.owner_id:
            raise ValidationError(_("The owner already has every kind of access."))
        if experiment.group_id is None or not self.user.memberships.filter(
                group_id=experiment.group_id).exists():
            raise ValidationError(
                _("%(user)s is not in this experiment's group, so it cannot be "
                  "shared with them.") % {"user": self.user.get_username()})


class ExperimentEvent(models.Model):
    """`ExperimentEvent` records one search-changing action, carried in the
    `.ihpo`. Each payload keeps whole before and after values, which lets any
    past state be rebuilt.
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
    #: Instance-only, like `Run.started_by`.
    actor = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="experiment_events")

    class Meta:
        ordering = ["at", "id"]


class BinEntry(models.Model):
    """`BinEntry` gives one entry per owner or contributor at deletion, because
    later grant changes must leave the bins alone. Viewers and uninvolved leads
    get none.
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
    """`Run` records a run's lifecycle so a background worker can drive it and
    pages can observe and cancel it.
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
    #: Who pressed Run; the worker checks the account's custom-model permission,
    #: since an unowned experiment has no owner to ask.
    started_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="runs_started")
    #: Every stopping criterion counts equally (STOPPING_CRITERIA).
    stopping = models.JSONField(default=dict)
    #: `trial_timeout` limits each model call with `{"mode", "seconds",
    #: "factor"}` and stays apart from stopping because it ends a trial. An empty
    #: value means the fixed default (`deadline.as_deadline`).
    trial_timeout = models.JSONField(default=dict, blank=True)
    #: The `priors` field copies the experiment's priors at creation, since
    #: `Experiment.priors` changes with every drag. `events` record what the
    #: prior did, and the `priors` field records what it was.
    priors = models.JSONField(default=dict, blank=True)
    #: Page-picked configurations, each `{"config", "origin", "proposal"}`, run
    #: first.
    configurations = models.JSONField(default=list, blank=True)
    #: Ending criterion; empty while running, cancelled or errored.
    stopped_by = models.CharField(max_length=40, blank=True, default="")
    primary_metric = models.CharField(max_length=100)
    #: With `trial_count`, the offset gives the range this run produced.
    trial_offset = models.IntegerField(null=True, blank=True)
    #: Search-level events such as a metric change; append-only.
    events = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    #: Each run stores its backend so the history survives a later configuration
    #: change and a restarted consumer can re-attach to the run's cluster job.
    backend = models.CharField(max_length=20, default="local")
    job_id = models.CharField(max_length=64, blank=True, default="")
    cancel_requested = models.BooleanField(default=False)
    #: `finished_at` follows once the trial in flight returns.
    cancel_requested_at = models.DateTimeField(null=True, blank=True)
    cancel_requested_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="runs_cancelled")
    error = models.TextField(blank=True, default="")
    #: Run press time; `started_at` may lag on a busy queue.
    created_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    #: Trial time in seconds; total minus this is overhead.
    trial_seconds = models.FloatField(null=True, blank=True)
    #: Trials performed; null until the run finishes.
    trial_count = models.IntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.experiment.title} run #{self.pk} ({self.status})"

    @property
    def max_trials(self):
        """`max_trials` gives the trial cap, or None for other bounds."""
        return self.stopping.get("max_trials")

    @property
    def duration(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class TrialModel(models.Model):
    """`TrialModel` keeps one ledger row per kept model, so storage sums rows
    instead of walking the disk. Each export refreshes `last_used_at`, which
    starts at the keeping time.
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
    """`GlobalSettings` keeps a single row, read through `get_solo()`."""

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
