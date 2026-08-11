from django.db import migrations

OLD, NEW = "SMAC (BlackBox)", "SMAC"


def rename(apps, schema_editor):
    """The optimizer was named after the one facade it was hardwired to.

    The facade is a setting now, so the name no longer describes the optimizer —
    an experiment searching with a random forest would still have said BlackBox.
    Stored rows are updated here; an `.ihpo` written before this still says the
    old name and is resolved through `SMACOptimizer.aliases` instead, because a
    file on someone else's disk cannot be migrated.
    """
    apps.get_model("ui", "Experiment").objects.filter(
        optimizer_name=OLD).update(optimizer_name=NEW)


def unrename(apps, schema_editor):
    apps.get_model("ui", "Experiment").objects.filter(
        optimizer_name=NEW).update(optimizer_name=OLD)


class Migration(migrations.Migration):

    dependencies = [
        ("ui", "0012_run_stopping_holds_every_criterion"),
    ]

    operations = [migrations.RunPython(rename, unrename)]
