"""`Group` and `Membership` draw the boundary that keeps research groups on one
instance from seeing each other's work. `Membership` records the member and lead
roles, per group. The site-admin role is the `manage_site` permission, which a
site admin can be granted without being a Django superuser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings as django_settings
from django.db import models

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
        from django.utils import timezone

        from experiments.models import Experiment

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
    ROLES = [(MEMBER, "Member"), (LEAD, "Group lead"),
             (PRIMARY_LEAD, "Primary group lead")]
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
    Django creates its `Permission` rows for the admin and `has_perm`. An active
    superuser has every one through `ModelBackend`.
    """

    id: int

    class Meta:
        managed = False
        default_permissions = ()
        permissions = [
            ("use_custom_models",
             "Can upload and run custom models (arbitrary code execution)"),
            ("manage_site",
             "Can manage groups, their size and their usage (a site admin)"),
            ("change_defaults",
             "Can change the experiment settings every experiment inherits"),
            # `view_all_experiments` and `manage_experiments` reach across every
            # group, so nobody has them by default; an admin grants them for a
            # support case and takes them back afterwards.
            ("view_all_experiments",
             "Can see every experiment on the instance, across all groups"),
            ("manage_experiments",
             "Can act on every experiment on the instance, across all groups"),
        ]
        verbose_name_plural = "Access permissions"
