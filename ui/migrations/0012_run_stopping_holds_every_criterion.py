from django.db import migrations, models


def carry_the_trial_cap_across(apps, schema_editor):
    """Move each existing run's `n_trials` into `stopping["max_trials"]`.

    Dropping the column without this would leave every past run looking as
    though it had no stopping criterion at all — which is now a state the code
    refuses, and would misreport the history of runs that completed perfectly
    well.
    """
    Run = apps.get_model("ui", "Run")
    for run in Run.objects.all().iterator():
        stopping = dict(run.stopping or {})
        stopping.setdefault("max_trials", run.n_trials)
        run.stopping = stopping
        run.save(update_fields=["stopping"])


def put_the_trial_cap_back(apps, schema_editor):
    Run = apps.get_model("ui", "Run")
    for run in Run.objects.all().iterator():
        run.n_trials = (run.stopping or {}).get("max_trials") or 0
        run.save(update_fields=["n_trials"])


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0011_experiment_cv_folds'),
    ]

    operations = [
        migrations.AlterField(
            model_name='run',
            name='stopping',
            field=models.JSONField(default=dict),
        ),
        migrations.RunPython(carry_the_trial_cap_across, put_the_trial_cap_back),
        migrations.RemoveField(
            model_name='run',
            name='n_trials',
        ),
    ]
