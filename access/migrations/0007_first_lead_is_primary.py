"""Give every group the lead who can appoint the others.

`0006` added the role; this fills it. A group's first lead becomes its primary
one — first by when they joined, which is the order in which the group was
actually built, and by primary key where two joined in the same transaction and
the timestamps tie.

A group with members but no lead at all gets its earliest member promoted
instead. That is a group nobody can manage, which is a worse state than one
whose primary lead was chosen by arithmetic: somebody has to be able to appoint
the rest, and the alternative is a site admin editing rows by hand before the
group can be used.

An empty group is left alone. There is nobody to promote, and the next person
added is the first lead by the same rule.

Reverse puts every primary lead back to an ordinary lead. It cannot restore
which of them was primary, because before this migration that was not recorded —
so it does not pretend to.
"""

from django.db import migrations


def forwards(apps, schema_editor):
    Group = apps.get_model("access", "Group")
    Membership = apps.get_model("access", "Membership")

    for group in Group.objects.all():
        if Membership.objects.filter(group=group, role="primary_lead").exists():
            continue
        members = Membership.objects.filter(group=group).order_by("joined_at", "pk")
        first = members.filter(role="lead").first() or members.first()
        if first is None:
            continue
        first.role = "primary_lead"
        first.save(update_fields=["role"])


def backwards(apps, schema_editor):
    Membership = apps.get_model("access", "Membership")
    Membership.objects.filter(role="primary_lead").update(role="lead")


class Migration(migrations.Migration):

    dependencies = [("access", "0006_alter_membership_role_and_more")]

    operations = [migrations.RunPython(forwards, backwards)]
