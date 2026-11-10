"""`Group` and `Membership` draw the boundary that keeps research groups on one
instance from seeing each other's work. `Membership` records the member and lead
roles, per group. Groups themselves are managed with `manage.py groups`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings as django_settings
from django.db import models
from django.utils.translation import gettext_lazy

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db.models.manager import RelatedManager

    from experiments.models import Experiment

#: `_PRIMARY_LEAD` lives at module scope because `Membership.Meta` cannot see
#: the class body's names.
_PRIMARY_LEAD = "primary_lead"


class Group(models.Model):
    """`Group` is the boundary experiments stay inside. Only a site admin edits
    `user_limit`, the seats a lead may fill, and `storage_limit_bytes`, where
    null means the site default. `is_active` lets a site admin mark a group
    restricted without deleting anything.
    """

    id: int
    memberships: RelatedManager[Membership]
    experiments: RelatedManager[Experiment]

    name = models.CharField(max_length=120, unique=True)
    user_limit = models.PositiveIntegerField(default=5)
    storage_limit_bytes = models.PositiveBigIntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    @property
    def storage_limit(self) -> int:
        """`storage_limit` prefers the group's limit to the site default."""
        if self.storage_limit_bytes is not None:
            return self.storage_limit_bytes
        return django_settings.GROUP_STORAGE_LIMIT_BYTES

    @property
    def member_count(self) -> int:
        return self.memberships.count()

    def delete(self, *args, **kwargs):
        """`delete` moves the group's experiments to the site trash for a site
        admin to rehome or discard, since nobody is left to judge whether the
        results matter. Otherwise `PROTECT` on `.Experiment.group` would block
        the deletion. A queryset delete skips `delete` and raises
        `ProtectedError` for a group that still contains experiments.
        """
        from django.db import transaction
        from django.utils import timezone

        from experiments.models import Experiment

        # Both steps share one transaction, so a group that `PROTECT` keeps also
        # keeps its work.
        with transaction.atomic():
            Experiment.objects.filter(group=self).update(
                group=None, trashed_at=timezone.now())
            return super().delete(*args, **kwargs)

    @property
    def seats_left(self) -> int:
        """`seats_left` stays at 0 for a group over a lowered limit."""
        return max(0, self.user_limit - self.member_count)


class Membership(models.Model):
    """`Membership` has one row per person and group, and the role lives here
    because a person may lead one group and only belong to another.
    """

    id: int
    user_id: int
    group_id: int
    get_role_display: Callable[[], str]

    MEMBER = "member"
    LEAD = "lead"
    #: A primary lead is a lead who also appoints and removes leads. The
    #: constraint in `Meta` keeps it to one per group.
    PRIMARY_LEAD = _PRIMARY_LEAD
    ROLES = [(MEMBER, gettext_lazy("Member")), (LEAD, gettext_lazy("Group lead")),
             (PRIMARY_LEAD, gettext_lazy("Primary group lead"))]
    #: Roles with a lead's reach over the group.
    LEAD_ROLES = (LEAD, PRIMARY_LEAD)

    user = models.ForeignKey(
        django_settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="memberships")
    #: `PROTECT` keeps a group with members from being deleted by accident.
    group = models.ForeignKey(Group, on_delete=models.PROTECT,
                              related_name="memberships")
    role = models.CharField(max_length=20, choices=ROLES, default=MEMBER)
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["group__name", "user__username"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "group"],
                name="one_membership_per_person_per_group"),
            # The database enforces one primary lead per group, so the admin and
            # data migrations cannot create a second one either.
            models.UniqueConstraint(
                fields=["group"],
                condition=models.Q(role=_PRIMARY_LEAD),
                name="one_primary_lead_per_group"),
        ]

    def __str__(self) -> str:
        return f"{self.user} in {self.group} ({self.get_role_display()})"

    @property
    def is_lead(self) -> bool:
        """`is_lead` is True for the primary lead too; every lead rule reads
        `is_lead`, and only `is_primary_lead` carries the extra power.
        """
        return self.role in self.LEAD_ROLES

    @property
    def is_primary_lead(self) -> bool:
        return self.role == self.PRIMARY_LEAD


class AccessPermissions(models.Model):
    """`AccessPermissions` is an unmanaged model with no table, present only so
    Django creates the `Permission` row behind `has_perm`. `manage.py users
    custom-models` grants it.
    """

    id: int

    class Meta:
        managed = False
        default_permissions = ()
        permissions = [
            ("use_custom_models",
             "Can upload and run custom models (arbitrary code execution)"),
        ]
        verbose_name_plural = "Access permissions"


class Account(models.Model):
    """`Account` stores what codesigner adds to a Django user: how the first
    password gets set, and whether a new one is required. `account_of` creates
    it on first use, so users made before it existed have one too.
    """

    #: The person chooses a password the first time they sign in by username.
    FIRST_SIGN_IN = "sign_in"
    #: The person sets a password through a `SetupLink`.
    SETUP_LINK = "link"
    PENDING = [(FIRST_SIGN_IN, "At first sign-in"), (SETUP_LINK, "Through a setup link")]

    id: int

    user = models.OneToOneField(django_settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="account")
    #: `password_pending` is empty once a password is set, and otherwise names
    #: how the person will set one.
    password_pending = models.CharField(max_length=10, choices=PENDING, blank=True, default="")
    #: Set by `manage.py users require-new-password`; cleared by a change.
    must_change_password = models.BooleanField(default=False)
    password_changed_at = models.DateTimeField(null=True, blank=True)
    #: `display_name` is what other people see in place of the email address.
    #: Names may repeat, and sign-in only uses the address. An empty name shows
    #: the address (`shown_name`).
    display_name = models.CharField(max_length=150, blank=True, default="")


#: The fewest characters a display name may have.
DISPLAY_NAME_MIN = 3


def shown_name(user) -> str:
    """`shown_name` is how *user* appears to other people: their display
    name, else their email address, else the account's internal name."""
    if user is None:
        return ""
    try:
        name = user.account.display_name
    except (Account.DoesNotExist, AttributeError):
        name = ""
    return name or user.email or user.get_username()


def account_of(user) -> Account:
    """`account_of` returns *user*'s `Account`, creating it when missing."""
    try:
        return user.account
    except Account.DoesNotExist:
        account, _ = Account.objects.get_or_create(user=user)
        return account


class SetupLink(models.Model):
    """`SetupLink` is a one-time link to set a password, stored as the SHA-256
    of its token, so the database never contains a usable link. It works until
    `expires_at` and only once (`used_at`)."""

    id: int

    user = models.ForeignKey(django_settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="setup_links")
    token_hash = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)


class AgreementAcceptance(models.Model):
    """`AgreementAcceptance` records that a person accepted one version of the
    agreement `ACCOUNT_AGREEMENT` supplies; a new version asks again."""

    id: int

    user = models.ForeignKey(django_settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="agreements")
    version = models.CharField(max_length=64)
    accepted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "version"],
                                               name="one_acceptance_per_version")]
