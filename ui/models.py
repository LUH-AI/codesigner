import secrets

from django.conf import settings as django_settings
from django.db import models, transaction
from django.utils.translation import gettext_lazy as _

from .fields import SafeJSONField


def generate_identifier() -> str:
    """A short, stable handle shown beside an experiment's name.

    Eight hex characters — enough to tell similarly-named experiments apart at
    a glance and to reference one independently of its (mutable) name.
    """
    return secrets.token_hex(4)


class ExperimentData(models.Model):
    """The experiment itself — everything an `.ihpo` file carries, and nothing else.

    One half of a deliberate split. This table holds what describes the *work*:
    what was searched, how it was evaluated, what was found. It is portable —
    every field here has a place in the snapshot `ui/services/snapshot.py`
    writes, and an `.ihpo` opened on another instance rebuilds exactly this row.

    `Experiment` holds the other half: who owns it, which group it is in, who it
    is shared with, what this instance was asked to draw. None of that travels.

    The split is a forcing function rather than a performance one. The boundary
    used to live only as prose in `snapshot_from_experiment`, so a field added
    to one long model joined the file or missed it depending on whether somebody
    remembered to add a line there. Now the table a field is declared in *is*
    the answer, and `tests/ui/experiments/test_the_two_halves.py` fails on a
    field that belongs to neither.
    """

    name = models.CharField(max_length=200)
    model_name = models.CharField(max_length=200)
    model_file = models.FileField(upload_to="custom_models/", blank=True, null=True)
    optimizer_name = models.CharField(max_length=200)
    optimizer_params = models.JSONField(default=dict, blank=True)
    metric_names = models.JSONField(default=list)
    # Which metric the optimizer is optimizing right now, vs. the one the
    # experiment's first run optimized (`original_metric`, pinned once and
    # never touched again). Named "current" rather than "primary" so the pair
    # reads as what it is — the two ends of a timeline — rather than "primary"
    # sounding like "the main one, as opposed to secondary metrics", which is
    # a different axis (see `metric_names`). The full history of every change
    # between them lives per-run, in `Run.primary_metric`/`Run.events`.
    current_metric = models.CharField(max_length=100, blank=True, null=True)
    original_metric = models.CharField(max_length=100, blank=True, null=True)
    seed = models.IntegerField(default=0)
    # How a trial is evaluated: 0 is a single holdout, k >= 2 is k-fold
    # cross-validation. Chosen when the experiment is created and fixed
    # thereafter — changing it mid-experiment would make the accumulated
    # trials incomparable with the new ones, the same fault a changed
    # metric used to have. See core.splits.
    cv_folds = models.IntegerField(default=0)
    # What share of the dataset is held out for validation, when the scheme is a
    # single split. Meaningless under cross-validation and simply carried, so
    # that switching an experiment's scheme would have somewhere to start from.
    test_size = models.FloatField(default=0.2)
    dataset = models.FileField(upload_to="datasets/", blank=True, null=True)
    # The search space the trials were drawn from, as ConfigSpace's own
    # serialized dict. Null until something supplies one.
    #
    # It is the one thing every surrogate-backed figure needs that `result` does
    # not carry, and it cannot always be asked for after the fact: a custom
    # model runs in its own process and a read-only rebuild never imports it, so
    # a page rendered later has nobody to ask. Filled from the model at the
    # moment a run has one (see services/run.py), or straight from the file for
    # an experiment read out of somebody else's run, which has no model at all.
    config_space = models.JSONField(blank=True, null=True, default=None)
    result = SafeJSONField(blank=True, null=True, default=None)
    # What the reader believes about where the optimum lies, by hyperparameter:
    # `{hp_name: {kind, params, exponent, decay, knots}}`. Stated on the
    # acquisition figure and applied to the next run (see
    # `SMACOptimizer._apply_priors`).
    #
    # `knots` is the density already evaluated on the grid, in ConfigSpace's
    # vectorized representation, and it is what actually reaches SMAC. The kind
    # and its parameters ride along only so the figure can put the controls back
    # where the reader left them. That split is the point: the density is
    # computed once, in the browser, and the server never re-implements it —
    # otherwise a Normal would mean one thing on the page and another in the
    # search, and the two would drift apart silently.
    #
    # SafeJSONField rather than JSONField for the reason that field exists: a
    # ruled-out region is a zero, and nothing stops a reader producing an
    # infinity by asking for a spike.
    priors = SafeJSONField(default=dict, blank=True)

    def __str__(self) -> str:
        return self.name


class ExperimentManager(models.Manager):
    """Creates both halves of an experiment, or neither.

    `Experiment.data` is `NOT NULL`, so the two rows have to be written
    together — and there is exactly one sensible order to write them in. Rather
    than leave every caller to remember it, `create()` takes the fields flat,
    sends each to the side that declares it, and does both in one transaction.

    This is a constructor, not a façade: reading still says `exp.data.result`,
    and a field is still declared on exactly one model. What it removes is the
    chance of a half-built experiment, which is the one failure the split could
    otherwise introduce.
    """

    def create(self, **fields):
        mine = {f.name for f in ExperimentData._meta.get_fields()
                if getattr(f, "concrete", False)}
        theirs = {key: fields.pop(key) for key in list(fields) if key in mine}
        with transaction.atomic():
            return super().create(data=ExperimentData.objects.create(**theirs),
                                  **fields)


class Experiment(models.Model):
    """This instance's record of an experiment — everything the file does not carry.

    Who owns it, which group it is in, what this installation was asked to
    compute and draw for it, and the state of the environment its model runs in.
    None of that means anything on another instance, which is exactly why it is
    not in the `.ihpo` — and now why it is not in `ExperimentData` either.

    The work itself is one hop away, through `data`. That hop is the point: a
    field added here is a claim that it is about *this instance*, and a field
    added there is a claim that it belongs in the file.
    """

    # The work this row is about. CASCADE because the two are one experiment
    # said twice: there is no meaning to an instance-side record of nothing.
    data = models.OneToOneField(ExperimentData, on_delete=models.CASCADE,
                                related_name="experiment")
    identifier = models.CharField(max_length=12, unique=True, editable=False, default=generate_identifier)
    created_at = models.DateTimeField(auto_now_add=True)
    # Per-experiment settings overrides; when use_default_settings is True the
    # global default experiment settings apply instead (see services/settings.py).
    #
    # Instance-side deliberately: a figure cap and an autocompute flag are what
    # *this* installation was asked to draw, and an `.ihpo` carried to another
    # one should arrive under that instance's own defaults.
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
    # Which group it is in, recorded here rather than read off the owner's
    # membership. The two answered the same question while every experiment had
    # an owner who was in a group, and they stop agreeing at exactly the moments
    # that matter: an account is deleted and `owner` goes null, or somebody
    # moves group and takes a colleague's work out of sight with them. The group
    # is the boundary, so the boundary is the thing stored.
    #
    # PROTECT, so a group cannot be deleted out from under work that is in it.
    # `Group.delete()` empties it into the bin first — see `access.models`.
    #
    # Null means one of two things, and `trashed_at` is what separates them:
    # set, the group was deleted under it and it is in the bin; unset, this is
    # an install with no accounts, where there are no groups to be in. Every
    # rule about the bin asks about `trashed_at`, never about this being empty.
    group = models.ForeignKey(
        "access.Group", null=True, blank=True,
        on_delete=models.PROTECT, related_name="experiments")
    # When its group was deleted under it. Set together with `group = None`, and
    # cleared by rehoming. A site admin decides what happens next; until they
    # do, the policy shows it to nobody, which is the same thing deletion would
    # have done except that it is reversible.
    trashed_at = models.DateTimeField(null=True, blank=True)
    # When somebody deleted it, and who. A different bin from `trashed_at`:
    # that one is a group's deletion, settled by a site admin; this is a
    # person's, settled by its people from their own bins (`BinEntry`). Nothing
    # is removed until one of them says so — the row, its runs, its grants and
    # its files all stay exactly as they were, so restoring is clearing this.
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="+")
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

    objects = ExperimentManager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    def refresh_from_db(self, *args, **kwargs):
        """Reload this row, and the half of the experiment that is not on it.

        Django reloads an instance's own columns and leaves cached relations
        alone, which is right in general and wrong here: `result` moved to
        `data`, and every caller that re-read an experiment to see what a run
        had written would have gone on holding the copy it fetched first.

        That is not a new hazard, it is the old one moved — before the split
        those columns were on this row and came back with it. Dropping the
        cache restores exactly that, so "refresh and read the result" keeps
        meaning what it has always meant.
        """
        self.__dict__.pop("data", None)
        return super().refresh_from_db(*args, **kwargs)

    @property
    def name(self) -> str:
        """The experiment's name, which lives with the search it names.

        The one field this side reads through `data` often enough to be worth a
        name of its own: `__str__`, the sidebar, every breadcrumb, the page
        title and the export's filename all ask for it.

        A property and not a second column, deliberately. A copy here would be
        a second representation of one fact, able to disagree with the file —
        which is the fault `_prior_record` in `services/snapshot.py` exists to
        avoid, and the reason `exponent` is not stored beside its schedule.
        Reading is free where the queryset said `select_related("data")`;
        *filtering* and *ordering* still say `data__name`, because those reach
        the database and the database knows only the column.
        """
        return self.data.name

    def save(self, *args, **kwargs):
        """Put a new experiment in its owner's group, when there is only one.

        The derivation survives exactly as far as it is unambiguous. Somebody in
        two groups has no "their group", and guessing would file their work
        under a boundary they did not choose — so there the caller says which,
        and the create form asks.

        Here rather than in the views that create one, because there are several
        — the create page, an imported `.ihpo`, the demo seed — and "every
        experiment is in a group" is a property of the model, not a step each of
        them has to remember. Only on the way in: afterwards the field is the
        record, and an owner who later changes group does not drag their old
        work across the boundary with them.

        Silent with no membership at all, which is an install with no accounts:
        there are no groups for it to be in, and inventing one here would be
        inventing the boundary as well.
        """
        if self._state.adding and self.group_id is None and self.owner_id:
            groups = list(self.owner.memberships.values_list("group_id", flat=True)[:2])
            if len(groups) == 1:
                self.group_id = groups[0]
        return super().save(*args, **kwargs)

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


class ExperimentShare(models.Model):
    """One person's access to somebody else's experiment.

    Per person and never per group: "my group" is not a fixed set of people,
    and an experiment shared with a group would be shared with whoever joins it
    next without anybody having decided that. Here every grant names somebody.

    Only a member of the experiment's own group can hold one. Sharing is how
    work moves *within* a group's boundary, not across it, and a grant that
    outlived its holder's membership would be exactly that crossing — so
    leaving the group takes the grants in it with them (see `access/signals.py`),
    and the policy asks about membership as well, in case something removed one
    without going through that.
    """

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    LEVELS = [
        # Look at it and take a copy away: everything the page shows, and the
        # .ihpo that carries the same.
        (VIEWER, _("Viewer")),
        # And run it, change its settings and delete it — everything but decide
        # who else may, which stays with the owner.
        (CONTRIBUTOR, _("Contributor")),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,
                                   related_name="shares")
    user = models.ForeignKey(django_settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE,
                             related_name="experiment_shares")
    level = models.CharField(max_length=16, choices=LEVELS, default=VIEWER)
    granted_at = models.DateTimeField(auto_now_add=True)
    # Who last set it, which is the owner at the time: ownership moves, and a
    # grant made by a previous owner is still a grant.
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
            return  # the required-field errors say this better
        experiment = self.experiment
        if self.user_id == experiment.owner_id:
            raise ValidationError(_("The owner already has every kind of access."))
        if experiment.group_id is None or not self.user.memberships.filter(
                group_id=experiment.group_id).exists():
            raise ValidationError(
                _("%(user)s is not in this experiment's group, so it cannot be "
                  "shared with them.") % {"user": self.user.get_username()})


class BinEntry(models.Model):
    """A deleted experiment, in one person's bin.

    Written at the moment of deletion, one per person the work belonged to: the
    owner and every contributor. Recorded rather than worked out from the grants
    each time the bin is opened, because the grants can change afterwards and
    whose bin something landed in should not.

    Not a viewer's, who could never have deleted it, and not a lead's who
    deleted it without being one of those — seeing a colleague's work was never
    the same as it being theirs.
    """

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
    # How long any one call to the model may take, and how that number is
    # arrived at: {"mode": "fixed"|"predicted", "seconds": float, "factor":
    # float}. Not a stopping criterion and deliberately not stored among them —
    # nothing here ends the run, it ends a trial, and the collector would filter
    # it out anyway. Empty for runs that predate the field, which
    # `core.modelhost.deadline.as_deadline` reads as the fixed default.
    trial_timeout = models.JSONField(default=dict, blank=True)
    # The beliefs this run searched under, copied from the experiment when the
    # run was created.
    #
    # `Experiment.priors` is the live, editable statement — it moves as soon as
    # somebody drags the curve. This is what was actually true when the trials
    # were spent, which is a different fact and the only one a reader can go
    # back to. The per-run `events` already record what the prior *did*
    # (applied, skipped, with which decay); this records what it *was*, which
    # none of them carry and nothing else can reconstruct.
    #
    # Empty for runs that predate the field, and for runs that stated nothing.
    # The two are the same thing to every reader of it: nothing to go back to.
    priors = models.JSONField(default=dict, blank=True)
    # Which criterion ended it. Empty while running, and for a run that was
    # cancelled or errored rather than stopping on its own terms.
    stopped_by = models.CharField(max_length=40, blank=True, default="")
    primary_metric = models.CharField(max_length=100)
    # How many trials the experiment already had. With `trial_count` this gives
    # the range this run produced, which is what turns a flat list of trials
    # back into a history of runs.
    trial_offset = models.IntegerField(null=True, blank=True)
    # Things that happened to the search rather than to a trial — so far, the
    # optimized metric changing, which rescores the whole history and throws
    # away any fitted surrogate. Appended to, never rewritten.
    events = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    # Where this run executed, and what the scheduler called it while it did.
    #
    # Recorded per run rather than read from the setting, for two reasons. An
    # experiment can be resumed months later under a different arrangement, and
    # the history should say what actually happened rather than what is
    # configured now. And a consumer that restarts while a job is still on the
    # cluster needs the id to re-attach to it — without that the job runs on
    # with nobody listening, and the row is swept to "error" beside it.
    backend = models.CharField(max_length=20, default="local")
    job_id = models.CharField(max_length=64, blank=True, default="")
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
