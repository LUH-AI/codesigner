from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    """The experiment tables moved to the `experiments` app with their rows, so
    only Django's record of where these links point changes."""

    dependencies = [
        ("assistant", "0006_conversations_keep_a_data_brief"),
        ("experiments", "0001_initial"),
    ]
    # `ui` forgets its copy of `Experiment` in 0045, so the links leave first.
    run_before = [("ui", "0045_experiments_own_their_tables")]

    operations = [
        migrations.SeparateDatabaseAndState(state_operations=[
            migrations.AlterField(
                model_name="conversation", name="experiment",
                field=models.ForeignKey(blank=True, null=True,
                                        on_delete=django.db.models.deletion.SET_NULL,
                                        related_name="conversations",
                                        to="experiments.experiment")),
            migrations.AlterField(
                model_name="agentaction", name="experiment",
                field=models.ForeignKey(blank=True, null=True,
                                        on_delete=django.db.models.deletion.SET_NULL,
                                        related_name="+", to="experiments.experiment")),
        ]),
    ]
