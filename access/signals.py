"""Deleting a `Membership` drops the member's shares in the group's experiments.
The `post_delete` receiver also catches admin, account and queryset deletes.
"""

from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Membership


@receiver(post_delete, sender=Membership)
def _grants_leave_with_the_member(sender, instance, **kwargs):
    """Grants never cross a group boundary."""
    from experiments.models import ExperimentShare

    ExperimentShare.objects.filter(
        user_id=instance.user_id,
        experiment__group_id=instance.group_id).delete()
