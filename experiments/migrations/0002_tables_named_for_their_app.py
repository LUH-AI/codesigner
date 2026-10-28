
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('experiments', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='binentry',
            table=None,
        ),
        migrations.AlterModelTable(
            name='experiment',
            table=None,
        ),
        migrations.AlterModelTable(
            name='experimentdata',
            table=None,
        ),
        migrations.AlterModelTable(
            name='experimentevent',
            table=None,
        ),
        migrations.AlterModelTable(
            name='experimentshare',
            table=None,
        ),
        migrations.AlterModelTable(
            name='globalsettings',
            table=None,
        ),
        migrations.AlterModelTable(
            name='run',
            table=None,
        ),
        migrations.AlterModelTable(
            name='trialmodel',
            table=None,
        ),
    ]
