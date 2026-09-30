"""What leaving a group takes with it.

A receiver rather than a step in the view that removes somebody, because that
view is not the only route: the Django admin deletes memberships, and so does
deleting the account. `post_delete` fires for each of them, a queryset delete
included — Django gives up its fast path for a model with receivers.
"""

from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Membership


@receiver(post_delete, sender=Membership)
def _grants_leave_with_the_member(sender, instance, **kwargs):
    """A grant is access *within* a group. Out of the group, it is access
    across the boundary, which nothing is allowed to grant."""
    from ui.models import ExperimentShare

    ExperimentShare.objects.filter(
        user_id=instance.user_id,
        experiment__group_id=instance.group_id).delete()
