"""`GuardedUserAdmin` extends Django's `UserAdmin`, guarding against deletion of
users who still own experiments. `Experiment.owner` is `SET_NULL`, so an
ownerless experiment has nobody who may share or rename it, and one in no group
is out of reach without the escape-hatch permissions. The guard lives here
because `PROTECT` on `on_delete` would also forbid a deliberately ownerless row.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.db.models import Count

admin.site.unregister(User)


@admin.register(User)
class GuardedUserAdmin(UserAdmin):
    """`GuardedUserAdmin` guards experiment owners against deletion."""

    list_display = (*UserAdmin.list_display, "is_active", "experiment_count")

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            _experiments=Count("experiments"))

    @admin.display(ordering="_experiments", description="Experiments")
    def experiment_count(self, user) -> int:
        """The column shows owners before a bulk delete fails."""
        return user._experiments

    def has_delete_permission(self, request, obj=None):
        """`has_delete_permission` returns False for an experiment owner, which
        hides the button on the change form. On the changelist, Django cancels
        the whole bulk delete if any selected user is guarded.
        """
        if obj is not None and obj.experiments.exists():
            return False
        return super().has_delete_permission(request, obj)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        """`change_view` explains the missing delete button, which otherwise
        looks like a bug.
        """
        user = self.get_object(request, object_id)
        if user is not None and user.experiments.exists():
            self.message_user(request, _KEPT % {"who": user.get_username()},
                              level="WARNING")
        return super().change_view(request, object_id, form_url, extra_context)


#: Deactivating an account blocks sign-in and keeps its experiments owned.
_KEPT = (
    "%(who)s still owns experiments, so this account cannot be deleted — "
    "deleting it would leave that work in no group, where nobody can reach it. "
    "Clear the “Active” box below to stop them signing in while their "
    "experiments stay theirs, or reassign the experiments to somebody else "
    "first."
)
