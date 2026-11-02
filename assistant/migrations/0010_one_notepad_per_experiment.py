"""An experiment's notes become one Markdown document, with a page of its own;
the notes written so far are joined into it, oldest first."""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def join_notes(apps, schema_editor):
    Note = apps.get_model("assistant", "ExperimentNote")
    Notes = apps.get_model("assistant", "ExperimentNotes")
    texts = {}
    for note in Note.objects.order_by("id"):
        texts.setdefault(note.experiment_id, []).append(note.text)
    for experiment_id, found in texts.items():
        Notes.objects.create(experiment_id=experiment_id, text="\n\n".join(found))


class Migration(migrations.Migration):

    dependencies = [
        ('assistant', '0009_chats_visit_and_notes'),
        ('experiments', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ExperimentNotes',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(blank=True, default='')),
                ('by_assistant', models.BooleanField(default=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('conversation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to='assistant.conversation')),
                ('experiment', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='notepad', to='experiments.experiment')),
                ('updated_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RunPython(join_notes, migrations.RunPython.noop),
        migrations.DeleteModel(
            name='ExperimentNote',
        ),
    ]
