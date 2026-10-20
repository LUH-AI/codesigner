from django.conf import settings as django_settings
from django.db import models

from experiments.fields import SafeJSONField


class FigureLayout(models.Model):
    """How one reader has arranged one experiment's page — see `ui/layout.py`.

    Per reader because it is a reading of the work, not the work: two people
    looking at the same experiment want different figures to hand. Not in the
    `.ihpo` for the same reason. A reader who has never rearranged a page has
    no row and sees the site default.

    `user` is null on an install without accounts, where there is one reader;
    the partial constraint keeps that to one row per experiment, which the
    plain unique one cannot, since NULLs never collide.
    """

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
    """How one reader wants an experiment page arranged until they arrange it.

    Between their own arrangement of an experiment (`FigureLayout`) and the
    site's default (`ui/figure_layout.toml`): what an experiment they have
    not touched opens on. `user` is null on an install without accounts, where
    there is one reader and so one row, which `layout.save_user_default` keeps
    to one by updating it rather than adding another.
    """

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="default_figure_layout")
    layout = SafeJSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)


class ReaderPreferences(models.Model):
    """One reader's own choices that are not about any one experiment.

    `user` is null on an install without accounts, where there is one reader
    and one row, as for `DefaultFigureLayout`.
    """

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, null=True, blank=True,
                                on_delete=models.CASCADE, related_name="reader_preferences")
    # How an exported file carries times, unless an experiment says otherwise:
    # "absolute", "relative" or "none" (see experiments/services/timestamps.py).
    export_timestamps = models.CharField(max_length=10, default="relative")
