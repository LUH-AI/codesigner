"""Share with people, one at a time, at a level: viewer or contributor.

Replaces two coarse flags. `shared` meant "my whole group may read this" and
`shared_with` meant "these people may read this", and neither could say *let
them run it*, which is the distinction the work needs. Both become `viewer`
grants, because both were read-only: nobody gains a power they did not have.

`shared` becomes one grant per person who is in the group now. That is the one
place this is not a like-for-like translation — the flag also covered whoever
joined later — and it is deliberately so: nobody decided those future members
should see it, which is the reason per-group sharing is going away.

A `shared_with` entry for somebody outside the experiment's group is dropped. It
was an invitation across the boundary, and there is no longer any such thing to
translate it into. The count is printed so that whoever migrates knows to tell
those owners.
"""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def forwards(apps, schema_editor):
    Experiment = apps.get_model("ui", "Experiment")
    Share = apps.get_model("ui", "ExperimentShare")
    Membership = apps.get_model("access", "Membership")

    members = {}
    for group_id, user_id in Membership.objects.values_list("group_id", "user_id"):
        members.setdefault(group_id, set()).add(user_id)

    grants = set()
    dropped = 0
    for exp in Experiment.objects.exclude(group_id=None).filter(shared=True):
        grants |= {(exp.pk, user) for user in members.get(exp.group_id, ())
                   if user != exp.owner_id}
    Through = Experiment.shared_with.through
    for exp_id, user_id, group_id, owner_id in Through.objects.values_list(
            "experiment_id", "user_id", "experiment__group_id", "experiment__owner_id"):
        if user_id == owner_id:
            continue
        if user_id in members.get(group_id, ()):
            grants.add((exp_id, user_id))
        else:
            dropped += 1

    Share.objects.bulk_create(
        Share(experiment_id=exp, user_id=user, level="viewer")
        for exp, user in sorted(grants))
    if dropped:
        print(f"\n  {dropped} share(s) with somebody outside the experiment's "
              f"group were dropped: sharing now stays within a group.")


def backwards(apps, schema_editor):
    """Every grant back as a name on the list — read-only, which is all the
    list could say. `shared` stays off: which grants were once the group-wide
    flag is not recorded anywhere."""
    Experiment = apps.get_model("ui", "Experiment")
    Share = apps.get_model("ui", "ExperimentShare")
    Through = Experiment.shared_with.through
    Through.objects.bulk_create(
        Through(experiment_id=exp, user_id=user)
        for exp, user in Share.objects.values_list("experiment_id", "user_id"))


class Migration(migrations.Migration):

    dependencies = [
        ("ui", "0026_split_experiment_data"),
        ("access", "0009_every_account_is_found_by_email"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ExperimentShare",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name="ID")),
                ("level", models.CharField(
                    choices=[("viewer", "Viewer"), ("contributor", "Contributor")],
                    default="viewer", max_length=16)),
                ("granted_at", models.DateTimeField(auto_now_add=True)),
                ("experiment", models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name="shares", to="ui.experiment")),
                ("granted_by", models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name="+", to=settings.AUTH_USER_MODEL)),
                ("user", models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name="experiment_shares", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "constraints": [models.UniqueConstraint(
                    fields=("experiment", "user"),
                    name="one_share_per_person_per_experiment")],
            },
        ),
        migrations.RunPython(forwards, backwards),
        migrations.RemoveField(model_name="experiment", name="shared_with"),
        migrations.RemoveField(model_name="experiment", name="shared"),
    ]
