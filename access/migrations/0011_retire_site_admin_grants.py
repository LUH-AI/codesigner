"""Site administration leaves the web app for the command line and
codesigner-manage, so the grants that opened it go: `manage_site`,
`change_defaults`, `view_all_experiments` and `manage_experiments`. Deleting a
`Permission` row also removes it from every user and group.

A superuser holds every permission whatever its rows say, and staff opened the
Django admin; both flags are cleared on accounts that do research, meaning they
belong to a group or own experiments. The migration prints each change.
"""

from django.db import migrations

RETIRED = ("manage_site", "change_defaults", "view_all_experiments", "manage_experiments")


def retire(apps, schema_editor):
    Permission = apps.get_model("auth", "Permission")
    User = apps.get_model("auth", "User")
    retired = Permission.objects.filter(content_type__app_label="access", codename__in=RETIRED)
    for permission in retired:
        holders = sorted(permission.user_set.values_list("username", flat=True))
        if holders:
            print(f"\n  {permission.codename} withdrawn from: {', '.join(holders)}", end="")
    retired.delete()

    researchers = (User.objects.filter(is_staff=True) | User.objects.filter(is_superuser=True))
    for user in researchers.distinct():
        if user.memberships.exists() or user.experiments.exists():
            print(f"\n  {user.username}: staff and superuser flags cleared", end="")
            user.is_staff = user.is_superuser = False
            user.save(update_fields=["is_staff", "is_superuser"])


class Migration(migrations.Migration):

    dependencies = [
        ('access', '0010_storage'),
        ('auth', '0012_alter_user_first_name_max_length'),
        ('experiments', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='accesspermissions',
            options={'default_permissions': (), 'managed': False, 'permissions': [('use_custom_models', 'Can upload and run custom models (arbitrary code execution)')], 'verbose_name_plural': 'Access permissions'},
        ),
        migrations.RunPython(retire, migrations.RunPython.noop),
    ]
