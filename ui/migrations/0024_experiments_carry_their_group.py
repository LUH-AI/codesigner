"""Record each experiment's group on the experiment.

`0023` added the field; this fills it from where the answer used to be read —
the owner's membership. Nothing changes about who can see what: for every row
that has an owner in a group, the stored value is exactly what
`owner__membership__group` was computing.

What it buys is the two cases where the derivation was about to give a different
answer from the one anybody intends. An account is deleted and `owner` goes null
— the work is still the group's, but a derived group has nothing left to read.
Somebody moves to another group — their old work should stay where it was done,
and a derived group would walk it across the boundary behind them.

Rows with no owner, or an owner in no group, are left empty rather than guessed
at. On an install with no accounts that is every row, and it is correct: there
are no groups for them to be in. They are not the bin — `trashed_at` is what
says that, and nothing here sets it.

Reverse drops the values. The derivation still works wherever an owner is in a
group, which is the state this migration found.
"""

from django.db import migrations


def forwards(apps, schema_editor):
    Experiment = apps.get_model("ui", "Experiment")
    Membership = apps.get_model("access", "Membership")

    # One query for the mapping rather than one per experiment: an instance with
    # a few hundred experiments and a dozen people should not make a few hundred
    # round trips to learn a dozen facts.
    group_of = dict(Membership.objects.values_list("user_id", "group_id"))
    if not group_of:
        return

    for experiment in Experiment.objects.filter(
            owner__isnull=False, group__isnull=True).only("id", "owner_id"):
        group_id = group_of.get(experiment.owner_id)
        if group_id is not None:
            Experiment.objects.filter(pk=experiment.pk).update(group_id=group_id)


def backwards(apps, schema_editor):
    Experiment = apps.get_model("ui", "Experiment")
    Experiment.objects.update(group=None)


class Migration(migrations.Migration):

    dependencies = [
        ("ui", "0023_experiment_group_experiment_trashed_at"),
        ("access", "0007_first_lead_is_primary"),
    ]

    operations = [migrations.RunPython(forwards, backwards)]
