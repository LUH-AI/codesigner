"""Make an email address the way a person is found.

Two steps. First every account is given an address it does not share with any
other, then the database is told to keep it that way.

**The backfill.** Existing addresses are lower-cased in place, since that is the
form they are compared in. An account with no address, or with one another
account already claimed, is given a placeholder at `.invalid` — a top-level
domain RFC 2606 reserves precisely so that a stand-in can never reach a real
inbox. An operator replaces these; until then the account is findable by a
string that is obviously not an address rather than by one belonging to
somebody else.

The earliest account keeps a contested address. That is an arbitrary rule, but
it is a stable one, and the alternative — keeping neither — would leave a real
person unfindable by the address they actually use.

**The index.** `email` belongs to `django.contrib.auth.User`, whose migrations
this project does not own, so no field here can be declared unique. The two
textbook workarounds cost more than they save: copying Django's whole auth
migration tree into this repository and faking it forward on every existing
install, or patching the field at start-up and leaving `makemigrations`
permanently reporting a change to an app it cannot write to. A unique index,
added from this side, is what is left. Case-insensitive, because the address is
compared that way — `COLLATE NOCASE` on SQLite and `LOWER()` on PostgreSQL.

**Partial, over real addresses only.** An empty string is not an address, so it
is not held to be unique: Django's own `createsuperuser` allows a blank, and so
does the admin's user editor. Such an account simply cannot be found by email,
which is the honest consequence of not having one — the backfill above gave
every *existing* account an address, and every account this project creates
itself goes through `find_or_create_by_email`, which refuses a blank.
"""

from django.db import migrations

UNSET_DOMAIN = "unset.invalid"
INDEX = "access_user_email_unique"


def give_everyone_an_address(apps, schema_editor):
    User = apps.get_model("auth", "User")
    seen = set()
    for user in User.objects.order_by("pk").iterator():
        address = (user.email or "").strip().lower()
        if not address or address in seen:
            base = (user.username or f"user{user.pk}").lower()
            address = f"{base}@{UNSET_DOMAIN}"
            n = 2
            while address in seen:
                address, n = f"{base}-{n}@{UNSET_DOMAIN}", n + 1
        if address != user.email:
            User.objects.filter(pk=user.pk).update(email=address)
        seen.add(address)


def add_index(apps, schema_editor):
    vendor = schema_editor.connection.vendor
    if vendor == "sqlite":
        sql = (f'CREATE UNIQUE INDEX "{INDEX}" ON "auth_user" ("email" COLLATE NOCASE) '
               f"WHERE \"email\" <> ''")
    elif vendor == "postgresql":
        sql = (f'CREATE UNIQUE INDEX "{INDEX}" ON "auth_user" (LOWER("email")) '
               f"WHERE \"email\" <> ''")
    else:
        # Anything else gets a plain unique index. Addresses are stored
        # lower-cased by the only code path that writes them, so the
        # difference is a race that path already closes.
        sql = f"CREATE UNIQUE INDEX {INDEX} ON auth_user (email) WHERE email <> ''"
    schema_editor.execute(sql)


def drop_index(apps, schema_editor):
    schema_editor.execute(f'DROP INDEX IF EXISTS "{INDEX}"')


class Migration(migrations.Migration):

    dependencies = [
        ("access", "0008_a_person_may_be_in_several_groups"),
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.RunPython(give_everyone_an_address, migrations.RunPython.noop),
        migrations.RunPython(add_index, drop_index),
    ]
