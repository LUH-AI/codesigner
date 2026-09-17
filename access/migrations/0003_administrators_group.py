"""Give the powers `is_staff` used to carry a name, and a home.

Codesigner used to read `is_staff` for three things: seeing every experiment,
acting on any of them, and changing the instance-wide defaults. It now asks for
named permissions instead, so that administering accounts and reading everyone's
unpublished results stop being the same grant.

That would silently take those powers away from anyone who has them today, so
this hands them back — as an **Administrators** group rather than four
checkboxes per person, since a group is how Django spells a role and the next
administrator should be one assignment rather than four.

Superusers are deliberately left out: Django's `ModelBackend` grants an active
superuser every permission already, so adding them would be a no-op that implied
the group was doing the work.
"""

from django.db import migrations

GROUP = "Administrators"

#: What the group holds. `use_custom_models` is included because an
#: administrator who cannot run a custom model cannot reproduce a report about
#: one; `ALLOW_CUSTOM_MODELS=False` is still the instance-wide floor beneath it.
PERMISSIONS = (
    "view_all_experiments",
    "manage_experiments",
    "change_defaults",
    "use_custom_models",
)


def _permissions(apps):
    """The `Permission` rows for this app's declared permissions.

    `create_permissions` is run from a `post_migrate` handler, which fires after
    every migration in the run — so the rows for permissions declared one
    migration ago do not exist *yet* when this one executes. Calling it here is
    the documented way round that, and skipping it makes this migration fail
    with a `DoesNotExist` that points at nothing in particular.

    It is handed the *live* app registry rather than the migration's historical
    one, whose app configs are stubs with no `models_module` — which is what
    `create_permissions` checks before it does anything.
    """
    from django.apps import apps as installed
    from django.contrib.auth.management import create_permissions

    create_permissions(installed.get_app_config("access"), verbosity=0)

    Permission = apps.get_model("auth", "Permission")
    return Permission.objects.filter(
        content_type__app_label="access", codename__in=PERMISSIONS)


def forwards(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    User = apps.get_model("auth", "User")

    group, _ = Group.objects.get_or_create(name=GROUP)
    group.permissions.set(_permissions(apps))

    # Exactly the people who had these powers a moment ago, and no one else.
    for user in User.objects.filter(is_staff=True, is_superuser=False):
        user.groups.add(group)


def backwards(apps, schema_editor):
    """Remove the group, leaving the accounts alone.

    Deleting users, or clearing their other groups, would be destroying
    something this migration did not create.
    """
    apps.get_model("auth", "Group").objects.filter(name=GROUP).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("access", "0002_alter_accesspermissions_options"),
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
