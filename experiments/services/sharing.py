"""Who else may reach an experiment, and handing it to somebody."""

from django.db import transaction

from experiments.models import ExperimentShare


def set_access(exp, colleague, level, granted_by) -> None:
    """Give *colleague* *level* access to *exp*, or none at all ("none")."""
    if level == "none":
        exp.shares.filter(user=colleague).delete()
    else:
        ExperimentShare.objects.update_or_create(
            experiment=exp, user=colleague,
            defaults={"level": level, "granted_by": granted_by})


def transfer(exp, colleague) -> bool:
    """Make contributor *colleague* *exp*'s owner, leaving the previous owner a
    contributor. Both happen together or not at all, so there is never a
    moment with no owner or with two. False if *colleague* is not a
    contributor."""
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
