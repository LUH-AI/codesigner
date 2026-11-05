"""Each run gets its own `Job` rows, which take over the run's `backend` and
`job_id` (now `scheduler_id`). Every run that started gets one job with the
run's times, status and trial range; a run still pending gets none.
"""

import django.db.models.deletion
from django.db import migrations, models


def jobs_from_runs(apps, schema_editor):
    Run = apps.get_model("experiments", "Run")
    Job = apps.get_model("experiments", "Job")
    jobs = []
    for run in Run.objects.exclude(status="pending").iterator():
        jobs.append(Job(
            run_id=run.pk, backend=run.backend or "local", scheduler_id=run.job_id or "",
            status=run.status, submitted_at=run.created_at, started_at=run.started_at,
            finished_at=run.finished_at, trial_offset=run.trial_offset,
            trial_count=run.trial_count, trial_seconds=run.trial_seconds, error=run.error))
    Job.objects.bulk_create(jobs)


class Migration(migrations.Migration):

    dependencies = [
        ('experiments', '0008_prior_history_from_runs'),
    ]

    operations = [
        migrations.CreateModel(
            name='Job',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('backend', models.CharField(default='local', max_length=20)),
                ('scheduler_id', models.CharField(blank=True, default='', max_length=64)),
                ('status', models.CharField(choices=[('submitted', 'Submitted'), ('running', 'Running'), ('done', 'Done'), ('error', 'Error'), ('cancelled', 'Cancelled')], default='running', max_length=20)),
                ('submitted_at', models.DateTimeField(blank=True, null=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
                ('trial_offset', models.IntegerField(blank=True, null=True)),
                ('trial_count', models.IntegerField(blank=True, null=True)),
                ('trial_seconds', models.FloatField(blank=True, null=True)),
                ('error', models.TextField(blank=True, default='')),
                ('run', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='jobs', to='experiments.run')),
            ],
            options={
                'ordering': ['-pk'],
            },
        ),
        migrations.RunPython(jobs_from_runs, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name='run',
            name='backend',
        ),
        migrations.RemoveField(
            model_name='run',
            name='job_id',
        ),
    ]
