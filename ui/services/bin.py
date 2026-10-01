"""Deleting an experiment, and undoing it.

Two bins, one after the other.

**Its people's.** Deleting sets an experiment aside rather than destroying it.
It drops out of every list and every URL — `experiments()` does not return it —
and lands in the bins of the people it belonged to: its owner and its
contributors. Any of them can put it back exactly as it was, or take a copy away
as an `.ihpo`. Taking it out of *your* bin is a decision about you, not about
the work: it leaves everyone else's bin alone and gives up your access, so a
colleague restoring it later does not hand it back to you.

**The site's.** Once nobody's bin holds it — every one of its people took it
out, or `BIN_RETENTION_DAYS` passed — it is nobody's, and waits for a site admin
with the work a deleted group leaves behind. Nothing expires there. A site admin
puts it back into a group or destroys it, which is the only way anything
deleted is ever destroyed.

Restoring is exact because nothing was changed on the way in. The row, both
halves of it, its runs, its grants and its files all stay where they were;
deleting is a timestamp and a list of whose bin it is in, and restoring is
clearing both.
"""

from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from ..models import BinEntry, Experiment, ExperimentShare

#: A run in one of these is stopped when its experiment is deleted.
_ACTIVE = ("pending", "running")


def cutoff():
    """Anything deleted before this has left its people's bins."""
    return timezone.now() - timedelta(days=settings.BIN_RETENTION_DAYS)


def expires_at(exp):
    """When *exp* leaves its people's bins on its own."""
    return exp.deleted_at + timedelta(days=settings.BIN_RETENTION_DAYS)


def orphaned():
    """Deleted, and in nobody's bin any more: the site admins' to settle.

    Worked out when asked rather than moved by a scheduled job, so an instance
    whose worker is down still answers correctly — and so do the bins, which
    ask the same `cutoff()` from the other side.
    """
    held = BinEntry.objects.filter(experiment=OuterRef("pk"))
    return Experiment.objects.filter(
        Q(deleted_at__lt=cutoff()) | ~Exists(held), deleted_at__isnull=False)


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


def dismiss(exp, user):
    """Take *exp* out of *user*'s bin, and *user* out of the experiment.

    Everyone else's bin is left as it was. Their grant goes with their entry —
    or a colleague restoring it would put it back in front of somebody who had
    decided to be done with it. An owner keeps the field, so the site's bin can
    still say whose it was; that their entry is gone is what tells `restore`
    they gave it up.
    """
    with transaction.atomic():
        exp.bin_entries.filter(user=user).delete()
        exp.shares.filter(user=user).delete()


def restore(exp, by=None):
    """Put *exp* back exactly as it was deleted. Out of every bin at once:
    it is one experiment, and it is either deleted or it is not.

    Exactly, except for whoever gave it up. Their grants were dropped when they
    took it out of their bins, so they stay out. If the owner was one of them,
    whoever restores it becomes its owner — they are a contributor, which is
    who ownership can be handed to anyway, and the work needs somebody who
    decides who else may reach it.
    """
    with transaction.atomic():
        if (getattr(by, "is_authenticated", False) and exp.owner_id != by.pk
                and not exp.bin_entries.filter(user_id=exp.owner_id).exists()):
            exp.shares.filter(user=by).delete()
            exp.owner = by
        _put_back(exp)


def put_back(exp, group):
    """A site admin's restore, from the site's bin, into *group*.

    For an experiment whose group was deleted, and for one nobody kept. The
    owner of record is left as it was — including somebody who took it out of
    their bin, since putting it back is the site admin's decision, not theirs.
    Whoever's bin it expired from keeps their grant: they never gave it up.
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
    """Destroy *exp* for good — the real cascade, runs and files with it."""
    exp.delete()
