import secrets

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
    dataset = models.FileField(upload_to="datasets/", blank=True, null=True)
    result = SafeJSONField(blank=True, null=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def is_running(self) -> bool:
        """True while a run is pending or executing (drives the sidebar spinner)."""
        return self.runs.filter(status__in=["pending", "running"]).exists()


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
    n_trials = models.IntegerField()
    primary_metric = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    cancel_requested = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    # Sum of this run's trial evaluation durations (seconds); total − this is
    # the search/bookkeeping overhead. Null until the run finishes.
    trial_seconds = models.FloatField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.experiment.name} run #{self.pk} ({self.status})"

    @property
    def duration(self):
        """Total wall-clock seconds, or None until both timestamps are set."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
