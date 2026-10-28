"""`sharing` grants access and transfers ownership."""

from django.db import transaction

from experiments.models import ExperimentShare


def set_access(exp, colleague, level, granted_by) -> None:
    """Level "none" removes the share."""
    if level == "none":
        exp.shares.filter(user=colleague).delete()
    else:
        ExperimentShare.objects.update_or_create(
            experiment=exp, user=colleague,
            defaults={"level": level, "granted_by": granted_by})


def transfer(exp, colleague) -> bool:
    """`transfer` atomically exchanges owner and contributor, giving False for a
    non-contributor."""
    grant = exp.shares.filter(user=colleague, level=ExperimentShare.CONTRIBUTOR).first()
    if grant is None:
        return False
    previous = exp.owner
    with transaction.atomic():
        grant.delete()
        exp.owner = colleague
        exp.save(update_fields=["owner"])
        if previous is not None:
            ExperimentShare.objects.update_or_create(
                experiment=exp, user=previous,
                defaults={"level": ExperimentShare.CONTRIBUTOR,
                          "granted_by": colleague})
    return True
