import secrets

from django.conf import settings as django_settings
from django.db import models

from .fields import SafeJSONField


def generate_identifier() -> str:
    """A short, stable handle shown beside an experiment's name.

    Eight hex characters — enough to tell similarly-named experiments apart at
    a glance and to reference one independently of its (mutable) name.
    """
    return secrets.token_hex(4)


class Experiment(models.Model):
    """A saved experiment — the database mirror of an .ihpo snapshot.

    Everything needed to reproduce and display an experiment: its identity,
    the model/optimizer/metrics/seed it was configured with, the dataset and
    any custom-model file, and the latest optimization result (the
    runhistory-mirrored dict, or None if it has never run).
    """

    name = models.CharField(max_length=200)
    identifier = models.CharField(max_length=12, unique=True, editable=False, default=generate_identifier)
    model_name = models.CharField(max_length=200)
    model_file = models.FileField(upload_to="custom_models/", blank=True, null=True)
    optimizer_name = models.CharField(max_length=200)
    optimizer_params = models.JSONField(default=dict, blank=True)
    metric_names = models.JSONField(default=list)
    primary_metric = models.CharField(max_length=100, blank=True, null=True)
    original_metric = models.CharField(max_length=100, blank=True, null=True)
    seed = models.IntegerField(default=0)
    # How a trial is evaluated: 0 is a single holdout, k >= 2 is k-fold
    # cross-validation. Chosen when the experiment is created and fixed
    # thereafter — changing it mid-experiment would make the accumulated
    # trials incomparable with the new ones, the same fault a changed
    # metric used to have. See core.splits.
    cv_folds = models.IntegerField(default=0)
    dataset = models.FileField(upload_to="datasets/", blank=True, null=True)
    result = SafeJSONField(blank=True, null=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True)
    # Per-experiment settings overrides; when use_default_settings is True the
    # global default experiment settings apply instead (see services/settings.py).
    settings = models.JSONField(default=dict, blank=True)
    use_default_settings = models.BooleanField(default=True)

    # ── Who it belongs to (see access/policy.py) ─────────────────────────────
    # Null means nobody's, which is every experiment on an install with no
    # accounts and every experiment that predates them. SET_NULL because
    # removing a person from an instance must not destroy their results — an
    # operator can reassign an ownerless experiment in the admin.
    owner = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="experiments")
    # Readable by everyone who can sign in. Not writable by them: sharing is an
    # invitation to look, not to run, rename or delete.
    shared = models.BooleanField(default=False)

    # ── The custom model's environment (see services/modelenv.py) ────────────
    # Pinned per experiment: resolved and locked once, so every run uses the
    # same dependencies and only the first pays for resolving them.
    ENV_NONE = "none"            # a registry model — nothing to prepare
    ENV_PENDING = "pending"      # queued
    ENV_PREPARING = "preparing"  # resolving and downloading
    ENV_READY = "ready"          # locked and built
    ENV_FAILED = "failed"        # see env_error
    ENV_SKIPPED = "skipped"      # no uv here, so it runs in this process
    ENV_LEGACY = "legacy"        # predates environments; runs in this process
    ENV_STATUS_CHOICES = [
        (ENV_NONE, "Not required"), (ENV_PENDING, "Queued"),
        (ENV_PREPARING, "Preparing"), (ENV_READY, "Ready"),
        (ENV_FAILED, "Failed"), (ENV_SKIPPED, "No uv — runs in-process"),
        (ENV_LEGACY, "Predates environments"),
    ]
    env_status = models.CharField(max_length=20, choices=ENV_STATUS_CHOICES, default=ENV_NONE)
    env_error = models.TextField(blank=True, default="")
    # Whatever is worth knowing about the environment without adding a column
    # for each: declared dependencies, requires-python, the resolved interpreter,
    # the model class, the lock's digest.
    env_meta = models.JSONField(default=dict, blank=True)
    env_prepared_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def is_running(self) -> bool:
        """True while a run is pending or executing (drives the sidebar spinner)."""
        return self.runs.filter(status__in=["pending", "running"]).exists()

    @property
    def env_pending(self) -> bool:
        """True while an environment is queued or being built."""
        return self.env_status in (self.ENV_PENDING, self.ENV_PREPARING)

    @property
    def env_in_process(self) -> bool:
        """True when this model runs in the application's own interpreter.

        Either uv was missing when it was prepared, or the experiment predates
        environments entirely. Kept as two states because the page says
        different things about them, but they run identically.
        """
        return self.env_status in (self.ENV_SKIPPED, self.ENV_LEGACY)


class Run(models.Model):
    """One optimization run of an Experiment.

    Records the run's lifecycle so it can be driven and observed out of band.
    The status transitions and the cancel flag are exercised once runs execute
    in the background (a later step); for now a run is created and completed
    synchronously.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("error", "Error"),
        ("cancelled", "Cancelled"),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="runs")
    # Who pressed Run. Null with no accounts, and after that account is
    # deleted. The worker needs it to know whose custom-model code it is
    # about to execute, which the experiment's owner does not always answer
    # — an unowned experiment is run by whoever is looking at it.
    started_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="runs_started")
    # Every stopping criterion, on equal footing — see
    # core.optimizers.base.STOPPING_CRITERIA. A run needs at least one and may
    # have any combination; a trial cap is one of them, not the frame the others
    # hang off.
    stopping = models.JSONField(default=dict)
    # Which criterion ended it. Empty while running, and for a run that was
    # cancelled or errored rather than stopping on its own terms.
    stopped_by = models.CharField(max_length=40, blank=True, default="")
    primary_metric = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    cancel_requested = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    # Sum of this run's trial evaluation durations (seconds); total − this is
    # the search/bookkeeping overhead. Null until the run finishes.
    trial_seconds = models.FloatField(null=True, blank=True)
    # Number of trials this run actually performed (may be < n_trials if
    # cancelled or the space was exhausted). Null until the run finishes.
    trial_count = models.IntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.experiment.name} run #{self.pk} ({self.status})"

    @property
    def max_trials(self):
        """The trial cap, if this run has one. Read by the pages that report
        progress as "n of m"; None when the run is bounded some other way."""
        return self.stopping.get("max_trials")

    @property
    def duration(self):
        """Total wall-clock seconds, or None until both timestamps are set."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class GlobalSettings(models.Model):
    """Single-row app settings, including the default experiment settings that
    new/inheriting experiments fall back to. Use `GlobalSettings.get_solo()`."""

    default_experiment_settings = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name_plural = "Global settings"

    def __str__(self) -> str:
        return "Global settings"

    @classmethod
    def get_solo(cls) -> "GlobalSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
