"""`delete` sets an experiment aside in the bins of its owner and contributors,
who can restore it or export an `.ihpo`. A person who dismisses it also gives up
access.

Once no bin holds it, or `BIN_RETENTION_DAYS` pass, a site admin restores or
destroys it. Restoring is exact because the rows, runs, grants and files stay in
place.
"""

from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Q, Exists, OuterRef
from django.utils import timezone

from experiments.models import account, BinEntry, Experiment, ExperimentShare

#: Run statuses stopped on deletion.
_ACTIVE = ("pending", "running")


def cutoff():
    """Older deletions have left personal bins."""
    return timezone.now() - timedelta(days=settings.BIN_RETENTION_DAYS)


def expires_at(exp):
    return exp.deleted_at + timedelta(days=settings.BIN_RETENTION_DAYS)


def orphaned():
    """`orphaned` queries the site bin on each call and needs no scheduled job,
    which keeps the answer correct while the worker is down.
    """
    held = BinEntry.objects.filter(experiment=OuterRef("pk"))
    return Experiment.objects.filter(
        Q(deleted_at__lt=cutoff()) | ~Exists(held), deleted_at__isnull=False)


def its_people(exp):
    """Contributors count only while in the group, as in
    `access.policy.share_of`.
    """
    people = set(exp.shares.filter(level=ExperimentShare.CONTRIBUTOR,
                                   user__memberships__group_id=exp.group_id)
                 .values_list("user_id", flat=True))
    if exp.owner_id is not None:
        people.add(exp.owner_id)
    return people


def delete(exp, by=None):
    """A running run gets a cancel request; its finished trials stay, and a
    restore leaves it stopped.
    """
    with transaction.atomic():
        exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
        exp.deleted_at = timezone.now()
        exp.deleted_by = account(by)
        exp.save(update_fields=["deleted_at", "deleted_by"])
        BinEntry.objects.bulk_create(
            [BinEntry(experiment=exp, user_id=user) for user in its_people(exp)],
            ignore_conflicts=True)


def dismiss(exp, user):
    """`dismiss` drops the user's share along with the bin entry. An owner keeps
    the `owner` field; the missing entry tells `restore` the owner gave it up.
    """
    with transaction.atomic():
        exp.bin_entries.filter(user=user).delete()
        exp.shares.filter(user=user).delete()


def restore(exp, by=None):
    """`restore` clears every bin at once and leaves out anyone who dismissed
    the experiment. If the owner dismissed it, the restoring contributor becomes
    owner.
    """
    with transaction.atomic():
        if (by is not None and getattr(by, "is_authenticated", False) and exp.owner_id != by.pk
                and not exp.bin_entries.filter(user_id=exp.owner_id).exists()):
            exp.shares.filter(user=by).delete()
            exp.owner = by
        _put_back(exp)


def put_back(exp, group):
    """`put_back` keeps the owner of record, even one who dismissed the
    experiment, and users whose entries expired keep their grants.
    """
    with transaction.atomic():
        exp.group = group
        exp.trashed_at = None
        _put_back(exp)


def _put_back(exp):
    exp.bin_entries.all().delete()
    exp.deleted_at = None
    exp.deleted_by = None
    exp.save(update_fields=["owner", "group", "trashed_at",
                            "deleted_at", "deleted_by"])


def purge(exp):
    """`purge` deletes for real, cascading to runs and files."""
    from .storage import delete_experiment_files

    delete_experiment_files(exp)
