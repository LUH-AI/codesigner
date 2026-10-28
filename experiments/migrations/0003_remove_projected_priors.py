from django.db import migrations

PROJECTED_KINDS = ("projected_prior_stated", "projected_prior_edited", "projected_prior_withdrawn")


def forget_projected_priors(apps, schema_editor):
    apps.get_model("experiments", "ExperimentEvent").objects.filter(kind__in=PROJECTED_KINDS).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("experiments", "0002_tables_named_for_their_app"),
    ]

    operations = [
        migrations.RunPython(forget_projected_priors, migrations.RunPython.noop),
        migrations.RemoveField(model_name="experimentdata", name="projected_priors"),
        migrations.RemoveField(model_name="run", name="projected_priors"),
    ]
