"""Give the pre-group experiments a home, and retire what groups replace.

Three things, of which only the first is the feature; the other two are debts
this feature calls in.

**1. Ownerless experiments.** `owner IS NULL` used to mean *everyone's*, which
was right while the only such experiments were the ones predating accounts:
there was no owner whose wishes were being overridden, and hiding them would
have swallowed an operator's work the moment they turned accounts on. With
groups it is a leak by construction — every member of every group would see
them — so the rule is gone and these need an owner instead. They go to the first
site admin, in a group of their own.

That does put experiments inside a site admin's group, which is against the
grain of a role defined by not seeing anyone's work. It is a one-off
consequence of data that predates the role, and the alternative — hiding them —
is the thing the original rule was written to avoid.

**2. The Administrators group.** `0003` created it to carry the powers split out
of `is_staff`. Roles carry them now, so it is retired. Safe to simply delete
only because it can be *shown* to hold nobody; if it does, this refuses rather
than dropping somebody's grants silently, and an operator reassigns first.

**3. Nothing is granted `view_all_experiments` or `manage_experiments`.** They
are instance-wide and cut straight through the boundary the rest of this draws.
See `access/models.py`.
"""

from django.db import migrations

ADMINISTRATORS = "Administrators"


def forwards(apps, schema_editor):
    Group = apps.get_model("access", "Group")
    Membership = apps.get_model("access", "Membership")
    Experiment = apps.get_model("ui", "Experiment")
    AuthGroup = apps.get_model("auth", "Group")

    _retire_administrators(AuthGroup)

    orphans = Experiment.objects.filter(owner__isnull=True)
    if not orphans.exists():
        return

    admin = _first_site_admin(apps)
    if admin is None:
        # Nobody to give them to. Left exactly as they are rather than guessed
        # at: the policy now shows them to nobody, and an operator assigns an
        # owner in the admin once there is somebody to assign. Saying so is the
        # point — they are waiting, not lost.
        return

    group, _ = Group.objects.get_or_create(
        name=_group_name(Group, admin), defaults={"user_limit": 1})
    Membership.objects.get_or_create(
        user=admin, defaults={"group": group, "role": "lead"})
    orphans.update(owner=admin)


def _first_site_admin(apps):
    """Whoever this instance already treats as running it.

    A superuser first, because that is what `createsuperuser` makes and what a
    single-operator instance has. Failing that, anyone holding `manage_site`.
    """
    User = apps.get_model("auth", "User")
    root = User.objects.filter(is_superuser=True).order_by("pk").first()
    if root is not None:
        return root
    return (User.objects
            .filter(user_permissions__codename="manage_site",
                    user_permissions__content_type__app_label="access")
            .order_by("pk").first())


def _group_name(Group, user) -> str:
    """A group named after them, or after them and a number if that is taken."""
    base = user.get_username() if hasattr(user, "get_username") else user.username
    name, n = base, 2
    while Group.objects.filter(name=name).exists():
        name, n = f"{base}-{n}", n + 1
    return name


def _retire_administrators(AuthGroup) -> None:
    group = AuthGroup.objects.filter(name=ADMINISTRATORS).first()
    if group is None:
        return
    held_by = list(group.user_set.values_list("username", flat=True))
    if held_by:
        raise RuntimeError(
            f"the {ADMINISTRATORS!r} group still holds "
            f"{', '.join(held_by)}. Roles replace it — make each of them a "
            "group lead or a site admin first, then remove the group; this "
            "migration will not drop their permissions for them.")
    group.delete()


def backwards(apps, schema_editor):
    """Nothing to undo.

    Un-assigning the experiments would mean guessing which had an owner before,
    and the previous migration can recreate the Administrators group. Reversing
    is about the schema; this is about what the rows mean, and that does not
    reverse.
    """


class Migration(migrations.Migration):

    dependencies = [
        ("access", "0004_group_alter_accesspermissions_options_membership"),
        ("ui", "0021_experiment_shared_with"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
