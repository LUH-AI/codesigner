"""The experiment tables, taken over from the ui app as they are.

State only: the tables already exist under their ui_ names.
"""

import django.db.models.deletion
import experiments.fields
import experiments.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('ui', '0044_projected_priors'),
        ('access', '0010_storage'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(state_operations=[
            migrations.CreateModel(
                name='ExperimentData',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('name', models.CharField(blank=True, default='', max_length=200)),
                    ('created_at', models.DateTimeField(blank=True, null=True)),
                    ('time_basis', models.CharField(choices=[('absolute', 'absolute'), ('relative', 'relative'), ('none', 'none')], default='absolute', max_length=10)),
                    ('model_name', models.CharField(max_length=200)),
                    ('model_file', models.FileField(blank=True, null=True, upload_to='custom_models/')),
                    ('optimizer_name', models.CharField(max_length=200)),
                    ('optimizer_params', models.JSONField(blank=True, default=dict)),
                    ('metric_names', models.JSONField(default=list)),
                    ('current_metric', models.CharField(blank=True, max_length=100, null=True)),
                    ('original_metric', models.CharField(blank=True, max_length=100, null=True)),
                    ('seed', models.IntegerField(default=0)),
                    ('cv_folds', models.IntegerField(default=0)),
                    ('test_size', models.FloatField(default=0.2)),
                    ('time_column', models.CharField(blank=True, default='', max_length=255)),
                    ('time_gap', models.PositiveIntegerField(default=0)),
                    ('horizon', models.PositiveIntegerField(default=0)),
                    ('series_column', models.CharField(blank=True, default='', max_length=255)),
                    ('season', models.PositiveIntegerField(default=0)),
                    ('task', models.CharField(choices=[('classification', 'classification'), ('regression', 'regression')], default='classification', max_length=20)),
                    ('dataset', models.FileField(blank=True, null=True, upload_to='datasets/')),
                    ('demo_dataset', models.CharField(blank=True, default='', max_length=200)),
                    ('processing', models.JSONField(blank=True, default=dict)),
                    ('config_space', models.JSONField(blank=True, default=None, null=True)),
                    ('result', experiments.fields.SafeJSONField(blank=True, default=None, null=True)),
                    ('priors', experiments.fields.SafeJSONField(blank=True, default=dict)),
                    ('projected_priors', experiments.fields.SafeJSONField(blank=True, default=list)),
                ],
                options={
                    'db_table': 'ui_experimentdata',
                },
            ),
            migrations.CreateModel(
                name='GlobalSettings',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('default_experiment_settings', models.JSONField(blank=True, default=dict)),
                ],
                options={
                    'verbose_name_plural': 'Global settings',
                    'db_table': 'ui_globalsettings',
                },
            ),
            migrations.CreateModel(
                name='Experiment',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('identifier', models.CharField(default=experiments.models.generate_identifier, editable=False, max_length=12, unique=True)),
                    ('created_at', models.DateTimeField(auto_now_add=True)),
                    ('settings', models.JSONField(blank=True, default=dict)),
                    ('use_default_settings', models.BooleanField(default=True)),
                    ('trashed_at', models.DateTimeField(blank=True, null=True)),
                    ('deleted_at', models.DateTimeField(blank=True, null=True)),
                    ('draft', models.BooleanField(default=False)),
                    ('setup_saved', models.JSONField(blank=True, default=dict)),
                    ('env_status', models.CharField(choices=[('none', 'Not required'), ('pending', 'Queued'), ('preparing', 'Preparing'), ('ready', 'Ready'), ('failed', 'Failed'), ('skipped', 'No uv — runs in-process'), ('legacy', 'Predates environments')], default='none', max_length=20)),
                    ('env_error', models.TextField(blank=True, default='')),
                    ('env_meta', models.JSONField(blank=True, default=dict)),
                    ('env_prepared_at', models.DateTimeField(blank=True, null=True)),
                    ('deleted_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
                    ('group', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='experiments', to='access.group')),
                    ('owner', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='experiments', to=settings.AUTH_USER_MODEL)),
                    ('data', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='experiment', to='experiments.experimentdata')),
                ],
                options={
                    'db_table': 'ui_experiment',
                    'ordering': ['-created_at'],
                },
            ),
            migrations.CreateModel(
                name='Run',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('stopping', models.JSONField(default=dict)),
                    ('trial_timeout', models.JSONField(blank=True, default=dict)),
                    ('priors', models.JSONField(blank=True, default=dict)),
                    ('projected_priors', models.JSONField(blank=True, default=list)),
                    ('configurations', models.JSONField(blank=True, default=list)),
                    ('stopped_by', models.CharField(blank=True, default='', max_length=40)),
                    ('primary_metric', models.CharField(max_length=100)),
                    ('trial_offset', models.IntegerField(blank=True, null=True)),
                    ('events', models.JSONField(blank=True, default=list)),
                    ('status', models.CharField(choices=[('pending', 'Pending'), ('running', 'Running'), ('done', 'Done'), ('error', 'Error'), ('cancelled', 'Cancelled')], default='pending', max_length=20)),
                    ('backend', models.CharField(default='local', max_length=20)),
                    ('job_id', models.CharField(blank=True, default='', max_length=64)),
                    ('cancel_requested', models.BooleanField(default=False)),
                    ('cancel_requested_at', models.DateTimeField(blank=True, null=True)),
                    ('error', models.TextField(blank=True, default='')),
                    ('created_at', models.DateTimeField(blank=True, null=True)),
                    ('started_at', models.DateTimeField(blank=True, null=True)),
                    ('finished_at', models.DateTimeField(blank=True, null=True)),
                    ('trial_seconds', models.FloatField(blank=True, null=True)),
                    ('trial_count', models.IntegerField(blank=True, null=True)),
                    ('cancel_requested_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='runs_cancelled', to=settings.AUTH_USER_MODEL)),
                    ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='runs', to='experiments.experiment')),
                    ('started_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='runs_started', to=settings.AUTH_USER_MODEL)),
                ],
                options={
                    'db_table': 'ui_run',
                },
            ),
            migrations.CreateModel(
                name='ExperimentEvent',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('at', models.DateTimeField(blank=True, null=True)),
                    ('at_trial', models.IntegerField(default=0)),
                    ('kind', models.CharField(max_length=40)),
                    ('payload', models.JSONField(blank=True, default=dict)),
                    ('actor', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='experiment_events', to=settings.AUTH_USER_MODEL)),
                    ('data', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='history', to='experiments.experimentdata')),
                    ('run', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='history', to='experiments.run')),
                ],
                options={
                    'db_table': 'ui_experimentevent',
                    'ordering': ['at', 'id'],
                },
            ),
            migrations.CreateModel(
                name='BinEntry',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('binned_at', models.DateTimeField(auto_now_add=True)),
                    ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='bin_entries', to=settings.AUTH_USER_MODEL)),
                    ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='bin_entries', to='experiments.experiment')),
                ],
                options={
                    'db_table': 'ui_binentry',
                    'constraints': [models.UniqueConstraint(fields=('experiment', 'user'), name='one_bin_entry_per_person_per_experiment')],
                },
            ),
            migrations.CreateModel(
                name='ExperimentShare',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('level', models.CharField(choices=[('viewer', 'Viewer'), ('contributor', 'Contributor')], default='viewer', max_length=16)),
                    ('granted_at', models.DateTimeField(auto_now_add=True)),
                    ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='shares', to='experiments.experiment')),
                    ('granted_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
                    ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='experiment_shares', to=settings.AUTH_USER_MODEL)),
                ],
                options={
                    'db_table': 'ui_experimentshare',
                    'constraints': [models.UniqueConstraint(fields=('experiment', 'user'), name='one_share_per_person_per_experiment')],
                },
            ),
            migrations.CreateModel(
                name='TrialModel',
                fields=[
                    ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                    ('trial', models.PositiveIntegerField()),
                    ('directory', models.CharField(max_length=500)),
                    ('bytes', models.PositiveBigIntegerField(default=0)),
                    ('created_at', models.DateTimeField(auto_now_add=True)),
                    ('last_used_at', models.DateTimeField(auto_now_add=True)),
                    ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='trial_models', to='experiments.experiment')),
                    ('run', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='trial_models', to='experiments.run')),
                ],
                options={
                    'db_table': 'ui_trialmodel',
                    'constraints': [models.UniqueConstraint(fields=('experiment', 'trial'), name='one_kept_model_per_trial')],
                },
            ),
        ]),
    ]
