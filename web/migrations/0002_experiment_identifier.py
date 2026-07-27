import secrets

from django.db import migrations, models

import web.models


def backfill_identifiers(apps, schema_editor):
    """Give every existing experiment a unique identifier before the column
    becomes unique and non-null. Done as data migration rather than a single
    AddField default, which would assign one identical value to all rows."""
    Experiment = apps.get_model("web", "Experiment")
    seen = set()
    for exp in Experiment.objects.filter(identifier__isnull=True):
        token = secrets.token_hex(4)
        while token in seen:
            token = secrets.token_hex(4)
        seen.add(token)
        exp.identifier = token
        exp.save(update_fields=["identifier"])


class Migration(migrations.Migration):

    dependencies = [
        ("web", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="experiment",
            name="identifier",
            field=models.CharField(editable=False, max_length=12, null=True),
        ),
        migrations.RunPython(backfill_identifiers, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="experiment",
            name="identifier",
            field=models.CharField(
                default=web.models.generate_identifier,
                editable=False,
                max_length=12,
                unique=True,
            ),
        ),
    ]
