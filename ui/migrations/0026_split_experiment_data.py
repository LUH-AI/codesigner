"""Split an experiment in two: the work, and this instance's record of it.

`ExperimentData` is exactly what an `.ihpo` carries — what was searched, how it
was evaluated, and what came out. `Experiment` keeps what the file has never
carried: who owns it, which group draws its boundary, what this installation
was asked to draw, and how its model runs here.

The boundary existed before this migration, but only as prose in
`snapshot_from_experiment`: a field added to the one long model joined the file
or missed it depending on whether somebody remembered to add a line there. Now
the table a field is declared in *is* the answer.

Done in four steps rather than one because the link is `NOT NULL` and the rows
already exist. Nullable, filled, then tightened — so that a backfill which fails
leaves every experiment exactly as it found it, rather than half of them
pointing at nothing.

`result` is dropped first among the fifteen. Each `RemoveField` rebuilds the
table on SQLite, and `result` is the trial history — the fourteen rebuilds after
it copy a narrow table instead of that column twice over per column.
"""

import django.db.models.deletion
from django.db import migrations, models

import ui.fields

#: What moves. Named once so the backfill and the drop cannot disagree about it.
MOVED = ("name", "model_name", "optimizer_name", "optimizer_params",
         "metric_names", "current_metric", "original_metric", "seed",
         "cv_folds", "test_size", "config_space", "priors")


def carry_the_data_across(apps, schema_editor):
    """One `ExperimentData` per `Experiment`, holding what it was searching.

    The two `FileField`s are copied by *name*. A `FileField`'s value is a path
    relative to `MEDIA_ROOT`, so assigning it moves which row points at the
    file and touches no file at all — where copying would double the media tree
    and leave half of it behind on a failure.

    `result` and `priors` are `SafeJSONField` on both sides, and `apps.get_model`
    hands back the real field class from the migration state, so the sentinel
    that carries an infinity across JSON encodes and decodes symmetrically.
    Nothing is rounded on the way.
    """
    Experiment = apps.get_model("ui", "Experiment")
    ExperimentData = apps.get_model("ui", "ExperimentData")

    for exp in Experiment.objects.all().iterator(chunk_size=200):
        data = ExperimentData.objects.create(
            result=exp.result,
            dataset=exp.dataset.name or "",
            model_file=exp.model_file.name or "",
            **{field: getattr(exp, field) for field in MOVED},
        )
        Experiment.objects.filter(pk=exp.pk).update(data=data)


def carry_it_back(apps, schema_editor):
    """Exact, because nothing was lost on the way out."""
    Experiment = apps.get_model("ui", "Experiment")
    for exp in Experiment.objects.select_related("data").iterator(chunk_size=200):
        data = exp.data
        Experiment.objects.filter(pk=exp.pk).update(
            result=data.result,
            dataset=data.dataset.name or "",
            model_file=data.model_file.name or "",
            **{field: getattr(data, field) for field in MOVED},
        )


class Migration(migrations.Migration):

    dependencies = [("ui", "0025_run_priors")]

    operations = [
        migrations.CreateModel(
            name="ExperimentData",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=200)),
                ("model_name", models.CharField(max_length=200)),
                ("model_file", models.FileField(blank=True, null=True,
                                                upload_to="custom_models/")),
                ("optimizer_name", models.CharField(max_length=200)),
                ("optimizer_params", models.JSONField(blank=True, default=dict)),
                ("metric_names", models.JSONField(default=list)),
                ("current_metric", models.CharField(blank=True, max_length=100, null=True)),
                ("original_metric", models.CharField(blank=True, max_length=100, null=True)),
                ("seed", models.IntegerField(default=0)),
                ("cv_folds", models.IntegerField(default=0)),
                ("test_size", models.FloatField(default=0.2)),
                ("dataset", models.FileField(blank=True, null=True, upload_to="datasets/")),
                ("config_space", models.JSONField(blank=True, default=None, null=True)),
                ("result", ui.fields.SafeJSONField(blank=True, default=None, null=True)),
                ("priors", ui.fields.SafeJSONField(blank=True, default=dict)),
            ],
        ),
        # Nullable for exactly as long as the backfill takes.
        migrations.AddField(
            model_name="experiment",
            name="data",
            field=models.OneToOneField(
                null=True, on_delete=django.db.models.deletion.CASCADE,
                related_name="experiment", to="ui.experimentdata"),
        ),
        migrations.RunPython(carry_the_data_across, carry_it_back),
        # And `NOT NULL` from here on, which is what makes an experiment with
        # no search in it unrepresentable rather than merely unusual.
        migrations.AlterField(
            model_name="experiment",
            name="data",
            field=models.OneToOneField(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="experiment", to="ui.experimentdata"),
        ),
        migrations.RemoveField(model_name="experiment", name="result"),
        migrations.RemoveField(model_name="experiment", name="dataset"),
        migrations.RemoveField(model_name="experiment", name="model_file"),
        *[migrations.RemoveField(model_name="experiment", name=field)
          for field in MOVED],
    ]
