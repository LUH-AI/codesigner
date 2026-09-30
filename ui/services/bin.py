"""Deleting an experiment, and undoing it.

Deleting sets the experiment aside rather than destroying it. It drops out of
every list and every URL — `experiments()` does not return it — and lands in
the bins of the people it belonged to: its owner and its contributors. Any of
them can put it back exactly as it was, or take a copy away as an `.ihpo`; only
its owner can destroy it, because that is the one step there is no undoing.

Restoring is exact because nothing was changed on the way in. The row, both
halves of it, its runs, its grants and its files all stay where they were;
deleting is a timestamp and a list of whose bin it is in, and restoring is
clearing both.
"""

from django.db import transaction
from django.utils import timezone

from ..models import BinEntry, ExperimentShare

#: A run in one of these is stopped when its experiment is deleted.
_ACTIVE = ("pending", "running")


def its_people(exp):
    """Whose bin a deleted *exp* goes to: its owner and its contributors.

    Contributors still in its group, for the same reason a grant stops counting
    once its holder leaves — see `access.policy.share_of`.
    """
    people = set(exp.shares.filter(level=ExperimentShare.CONTRIBUTOR,
                                   user__memberships__group_id=exp.group_id)
                 .values_list("user_id", flat=True))
    if exp.owner_id is not None:
        people.add(exp.owner_id)
    return people


def delete(exp, by=None):
    """Set *exp* aside, into its people's bins.

    A run still going is asked to stop, as before — the trials it finished are
    kept, and restoring does not start it again.
    """
    with transaction.atomic():
        exp.runs.filter(status__in=_ACTIVE).update(cancel_requested=True)
        exp.deleted_at = timezone.now()
        exp.deleted_by = by if getattr(by, "is_authenticated", False) else None
        exp.save(update_fields=["deleted_at", "deleted_by"])
        BinEntry.objects.bulk_create(
            [BinEntry(experiment=exp, user_id=user) for user in its_people(exp)],
            ignore_conflicts=True)


def restore(exp):
    """Put *exp* back exactly as it was deleted. Out of every bin at once:
    it is one experiment, and it is either deleted or it is not."""
    with transaction.atomic():
        exp.bin_entries.all().delete()
        exp.deleted_at = None
        exp.deleted_by = None
        exp.save(update_fields=["deleted_at", "deleted_by"])


def purge(exp):
    """Destroy *exp* for good — the real cascade, runs and files with it."""
    exp.delete()
