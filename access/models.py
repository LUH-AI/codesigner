"""Who belongs to which group, and the permissions that are not about a row.

An instance hosted for several research groups exists so that the groups do
**not** see each other. That is what `Group` and `Membership` are for, and it is
the reason almost everything here is about *narrowing* rather than granting.

Three roles, and only one of them says "admin":

* **Member** — creates and runs experiments, and chooses for each whether it is
  shared with their group, with named colleagues, or with nobody.
* **Group lead** — a member who also manages their group's people, and can see
  and act on anything in it. A `Membership` with `role = LEAD`.
* **Site admin** — manages groups: how large, whether active, what they are
  using. Holds `manage_site`, belongs to no group, and is shown no experiments.

The first two are a membership because they are a fact about a person *within* a
group. The third is a permission because it is a fact about the instance, and
because it must be grantable without making somebody a Django superuser — see
the site panel on why that distinction carries weight.
"""

from django.conf import settings as django_settings
from django.db import models

#: `Membership.PRIMARY_LEAD`, at module scope because the constraint that keeps
#: it unique is declared in `Membership.Meta`, and a class body's own names are
#: not visible from there.
_PRIMARY_LEAD = "primary_lead"


class Group(models.Model):
    """A research group: the boundary experiments are not visible across.

    `user_limit` is the seats a group lead may fill. It is theirs to fill and
    not theirs to raise — a lead who could change it would be a lead with no
    limit — so only a site admin may edit it.

    `is_active` is how a site admin restricts a group without destroying
    anything: its members cannot sign in or start runs, and every experiment
    stays exactly where it was. Deletion is deliberately not the tool for that,
    and `Membership.group` is `PROTECT` so it cannot become one by accident.
    """

    name = models.CharField(max_length=120, unique=True)
    user_limit = models.PositiveIntegerField(default=5)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    @property
    def member_count(self) -> int:
        return self.memberships.count()

    def delete(self, *args, **kwargs):
        """Empty the group's work into the bin, then go.

        `Experiment.group` is PROTECT, so without this a group that still holds
        anything cannot be deleted at all. Taking the experiments down with it
        is the other obvious answer and the wrong one: a group is removed once
        its people have gone, which is exactly the moment nobody is left to say
        whether the results still matter. So they are set aside for a site admin
        to rehome or discard, and the deletion goes through.

        A queryset delete does not come through here and will raise
        `ProtectedError` instead, which is the safe way round: it refuses rather
        than quietly taking a different route.
        """
        from django.utils import timezone

        from ui.models import Experiment

        Experiment.objects.filter(group=self).update(
            group=None, trashed_at=timezone.now())
        return super().delete(*args, **kwargs)

    @property
    def seats_left(self) -> int:
        """How many more people may be added. Never negative.

        A limit lowered below the number of people already in the group leaves
        it full rather than in deficit: nobody is removed by arithmetic, and the
        lead simply cannot add more until somebody leaves.
        """
        return max(0, self.user_limit - self.member_count)


class Membership(models.Model):
    """Which group somebody is in, and what they are within it.

    One group per person — a OneToOne rather than a many-to-many — because
    "which group is this experiment in" has to have one answer. An experiment's
    group is its owner's, and that is the whole of how the boundary is drawn.
    """

    MEMBER = "member"
    LEAD = "lead"
    #: A lead with one thing more: leads are theirs to appoint and to unmake.
    #:
    #: A separate role rather than a flag on the group, because it is a fact
    #: about a person *within* a group in exactly the way the other two are, and
    #: the answer to "what is this person here" should come from one field. The
    #: constraint below is what makes it at most one.
    PRIMARY_LEAD = _PRIMARY_LEAD
    ROLES = [(MEMBER, "Member"), (LEAD, "Group lead"),
             (PRIMARY_LEAD, "Primary group lead")]
    #: The roles that carry a lead's reach over the group. Named rather than
    #: written out at each test, so adding a fourth role is one edit here.
    LEAD_ROLES = (LEAD, PRIMARY_LEAD)

    user = models.OneToOneField(
        django_settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="membership")
    # PROTECT: a group with people in it is not something to delete by way of
    # tidying up. Restricting it is `is_active`; emptying it is deliberate.
    group = models.ForeignKey(Group, on_delete=models.PROTECT,
                              related_name="memberships")
    role = models.CharField(max_length=20, choices=ROLES, default=MEMBER)
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["group__name", "user__username"]
        constraints = [
            # One per group, enforced in the database rather than in the views
            # that appoint one: a second primary lead is not a state the
            # application should be able to reach by any route, including the
            # admin and a data migration. Partial, so the other two roles are
            # unaffected — a group has many members and may have many leads.
            models.UniqueConstraint(
                fields=["group"],
                condition=models.Q(role=_PRIMARY_LEAD),
                name="one_primary_lead_per_group"),
        ]

    def __str__(self) -> str:
        return f"{self.user} in {self.group} ({self.get_role_display()})"

    @property
    def is_lead(self) -> bool:
        """Whether this membership carries a lead's reach over the group.

        True for the primary lead as well: they are a lead who can also appoint
        them, not a different thing that happens to sit above one. Every rule
        about what a lead may see and do reads this, so the extra power stays
        confined to `is_primary_lead` and the two views that ask for it.
        """
        return self.role in self.LEAD_ROLES

    @property
    def is_primary_lead(self) -> bool:
        return self.role == self.PRIMARY_LEAD


class AccessPermissions(models.Model):
    """Permissions that are not about a row.

    Properties of an account rather than of any particular experiment, so there
    is no model they naturally hang off. Django's answer is an unmanaged model
    with no default permissions: no table is created, but the `Permission` rows
    are, so each appears in the admin's user editor and
    `user.has_perm("access.…")` works.

    An active superuser holds all of them without being granted anything,
    because that is what Django's `ModelBackend` does with `has_perm`.
    """

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
            # ── the escape hatch ────────────────────────────────────────────
            # Instance-wide, and therefore straight through the group boundary
            # everything else here exists to draw. Granted to nobody, and meant
            # to be handed out for a support case and taken back afterwards —
            # not to describe a role. A group lead gets the same reach inside
            # their own group from their membership, which is the ordinary way.
            ("view_all_experiments",
             "Can see every experiment on the instance, across all groups"),
            ("manage_experiments",
             "Can act on every experiment on the instance, across all groups"),
        ]
        verbose_name_plural = "Access permissions"
