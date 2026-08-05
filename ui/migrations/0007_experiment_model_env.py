from django.db import migrations, models


def mark_existing_as_legacy(apps, schema_editor):
    """Custom-model experiments that predate environments keep working.

    They have no runner and no lock, so they run in this process exactly as they
    always did, and the page offers to build them an environment. Marking them
    `pending` instead would start an unattended resolve for every one of them on
    the next worker start — mostly on files with no PEP 723 header, so mostly
    failing. An upgrade should not begin a download or break what worked.
    """
    Experiment = apps.get_model("ui", "Experiment")
    Experiment.objects.exclude(model_file="").exclude(model_file=None).update(
        env_status="legacy")


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0006_run_trial_count'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='env_error',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='experiment',
            name='env_meta',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='experiment',
            name='env_prepared_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='env_status',
            field=models.CharField(choices=[('none', 'Not required'), ('pending', 'Queued'), ('preparing', 'Preparing'), ('ready', 'Ready'), ('failed', 'Failed'), ('skipped', 'No uv — runs in-process'), ('legacy', 'Predates environments')], default='none', max_length=20),
        ),
        migrations.RunPython(mark_existing_as_legacy, migrations.RunPython.noop),
    ]
