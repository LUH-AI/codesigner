from __future__ import annotations

from django.conf import settings as django_settings
from django.db import models

from experiments.fields import SafeJSONField


class FigureLayout(models.Model):
    """`FigureLayout` stores one reader's arrangement of one experiment's page.
    The partial constraint covers a null `user`, since NULLs never collide in a
    plain unique constraint.
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
    """`DefaultFigureLayout` overrides `ui/figure_layout.toml` per reader.
    """

    id: int
    user_id: int | None

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="default_figure_layout")
    layout = SafeJSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)


class ReaderPreferences(models.Model):

    id: int
    user_id: int | None

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="reader_preferences")
    export_timestamps = models.CharField(max_length=10, default="relative")
    #: The reader's own switch; the instance's `ASSISTANT_ENABLED` comes first.
    assistant_enabled = models.BooleanField(default=True)
