from __future__ import annotations

from django.conf import settings as django_settings
from django.db import models

from experiments.fields import SafeJSONField


class FigureLayout(models.Model):
    """`FigureLayout` stores one reader's arrangement of one experiment's page
    and stays out of the `.ihpo`. A reader with no row sees the site default.
    With `user` null on an install without accounts, the partial constraint
    keeps one row per experiment, since NULLs never collide in the plain unique
    one.
    """

    id: int
    experiment_id: int
    user_id: int | None

    experiment = models.ForeignKey("experiments.Experiment", on_delete=models.CASCADE,
                                   related_name="figure_layouts")
    user = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.CASCADE, related_name="figure_layouts")
    layout = SafeJSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["experiment", "user"],
                                    name="one_layout_per_reader_per_experiment"),
            models.UniqueConstraint(fields=["experiment"],
                                    condition=models.Q(user__isnull=True),
                                    name="one_layout_per_experiment_without_accounts"),
        ]


class DefaultFigureLayout(models.Model):
    """`DefaultFigureLayout` arranges experiments the reader has no
    `FigureLayout` for, ahead of `ui/figure_layout.toml`.
    """

    id: int
    user_id: int | None

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="default_figure_layout")
    layout = SafeJSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)


class ReaderPreferences(models.Model):
    """`ReaderPreferences` stores per-reader choices beyond any one experiment.
    """

    id: int
    user_id: int | None

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="reader_preferences")
    export_timestamps = models.CharField(max_length=10, default="relative")
