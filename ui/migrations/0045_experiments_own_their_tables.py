"""The experiment tables now belong to the experiments app.

State only, and the content types follow, so admin history and permissions
keep pointing at the same rows.
"""

import django.db.models.deletion
from django.db import migrations, models


MOVED = ("experimentdata", "experiment", "experimentshare", "experimentevent",
         "binentry", "run", "trialmodel", "globalsettings")


def relabel(app_label_from, app_label_to):
    def run(apps, schema_editor):
        ContentType = apps.get_model("contenttypes", "ContentType")
        ContentType.objects.filter(app_label=app_label_from, model__in=MOVED).update(
            app_label=app_label_to)
    return run


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('experiments', '0001_initial'),
        ('ui', '0044_projected_priors'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(state_operations=[
            migrations.RemoveConstraint(
                model_name='binentry',
                name='one_bin_entry_per_person_per_experiment',
            ),
            migrations.RemoveField(
                model_name='experiment',
                name='data',
            ),
            migrations.RemoveField(
                model_name='experiment',
                name='deleted_by',
            ),
            migrations.RemoveField(
                model_name='experiment',
                name='group',
            ),
            migrations.RemoveField(
                model_name='experiment',
                name='owner',
            ),
            migrations.AlterField(
                model_name='figurelayout',
                name='experiment',
                field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='figure_layouts', to='experiments.experiment'),
            ),
            migrations.RemoveField(
                model_name='run',
                name='experiment',
            ),
            migrations.RemoveField(
                model_name='experimentevent',
                name='data',
            ),
            migrations.RemoveField(
                model_name='experimentevent',
                name='actor',
            ),
            migrations.RemoveField(
                model_name='experimentevent',
                name='run',
            ),
            migrations.RemoveConstraint(
                model_name='experimentshare',
                name='one_share_per_person_per_experiment',
            ),
            migrations.DeleteModel(
                name='GlobalSettings',
            ),
            migrations.RemoveField(
                model_name='run',
                name='cancel_requested_by',
            ),
            migrations.RemoveField(
                model_name='run',
                name='started_by',
            ),
            migrations.RemoveConstraint(
                model_name='trialmodel',
                name='one_kept_model_per_trial',
            ),
            migrations.RemoveField(
                model_name='binentry',
                name='experiment',
            ),
            migrations.RemoveField(
                model_name='binentry',
                name='user',
            ),
            migrations.DeleteModel(
                name='ExperimentData',
            ),
            migrations.DeleteModel(
                name='ExperimentEvent',
            ),
            migrations.RemoveField(
                model_name='experimentshare',
                name='experiment',
            ),
            migrations.RemoveField(
                model_name='experimentshare',
                name='granted_by',
            ),
            migrations.RemoveField(
                model_name='experimentshare',
                name='user',
            ),
            migrations.RemoveField(
                model_name='trialmodel',
                name='experiment',
            ),
            migrations.RemoveField(
                model_name='trialmodel',
                name='run',
            ),
            migrations.DeleteModel(
                name='BinEntry',
            ),
            migrations.DeleteModel(
                name='ExperimentShare',
            ),
            migrations.DeleteModel(
                name='Experiment',
            ),
            migrations.DeleteModel(
                name='Run',
            ),
            migrations.DeleteModel(
                name='TrialModel',
            ),
        ]),
        migrations.RunPython(relabel("ui", "experiments"), relabel("experiments", "ui")),
    ]
