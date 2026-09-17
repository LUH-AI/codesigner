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
    ROLES = [(MEMBER, "Member"), (LEAD, "Group lead")]

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

    def __str__(self) -> str:
        return f"{self.user} in {self.group} ({self.get_role_display()})"

    @property
    def is_lead(self) -> bool:
        return self.role == self.LEAD


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
