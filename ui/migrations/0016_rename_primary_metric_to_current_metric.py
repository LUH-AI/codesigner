from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0015_remove_run_optimizer_params'),
    ]

    operations = [
        migrations.RenameField(
            model_name='experiment',
            old_name='primary_metric',
            new_name='current_metric',
        ),
    ]
