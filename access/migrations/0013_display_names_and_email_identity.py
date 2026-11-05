"""People are identified by their email address and shown by a display name
(`Account.display_name`), so the username becomes a hidden copy of the
address: every account with an address gets it as its username. An account
whose address another account already uses as its username, or which has no
address, keeps its old username and signs in with that. The migration prints
each account it leaves alone.
"""

from django.db import migrations, models


def usernames_follow_addresses(apps, schema_editor):
    User = apps.get_model("auth", "User")
    for user in User.objects.all():
        address = (user.email or "").strip().lower()
        if ":" in user.username or user.username == address:
            continue
        if not address:
            print(f"\n  {user.username}: no email address, keeps signing in by this name", end="")
            continue
        if User.objects.filter(username__iexact=address).exclude(pk=user.pk).exists():
            print(f"\n  {user.username}: {address} is another account's username; left as is",
                  end="")
            continue
        user.username = address
        user.save(update_fields=["username"])


class Migration(migrations.Migration):

    dependencies = [
        ('access', '0012_accounts_and_setup_links'),
    ]

    operations = [
        migrations.AddField(
            model_name='account',
            name='display_name',
            field=models.CharField(blank=True, default='', max_length=150),
        ),
        migrations.RunPython(usernames_follow_addresses, migrations.RunPython.noop),
    ]
