"""The user editor, with one thing taken away.

Django's `UserAdmin` is the user management for this application — creating
accounts, setting passwords, granting permissions, assigning groups. None of
that is reimplemented; this replaces it only to stop one action that is quietly
destructive.

**Deleting a user makes their work unreachable.** `Experiment.owner` is
`SET_NULL`, so the rows survive — and an experiment belonging to nobody is in no
group, which under `access.policy.GroupPolicy` means nobody can see it. The work
does not go away; it goes silent, and the only route back is this admin.

Worth knowing that this guard predates groups, when the same deletion had the
opposite consequence and a worse one: ownerless used to mean *everyone's*, so
deleting a departing colleague published their unpublished work across the whole
instance. Groups closed that. What did not change is that an administrator
tidying up accounts does not expect either outcome, which is why the refusal is
still here.

Enforced at the only place users are deleted rather than by changing
`on_delete`: `PROTECT` would make the constraint real at the database, but it
would also forbid a deliberately ownerless row and surface as an
`IntegrityError` somewhere with no idea why.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.db.models import Count

admin.site.unregister(User)


@admin.register(User)
class GuardedUserAdmin(UserAdmin):
    """Django's user editor, refusing to delete anyone who owns experiments."""

    list_display = UserAdmin.list_display + ("is_active", "experiment_count")

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            _experiments=Count("experiments"))

    @admin.display(ordering="_experiments", description="Experiments")
    def experiment_count(self, user) -> int:
        """Shown in the list so the state is visible *before* anyone reaches for
        delete, rather than only in the refusal afterwards."""
        return user._experiments

    def has_delete_permission(self, request, obj=None):
        """False for a user who owns experiments.

        On the change form this hides the button, so nobody is walked through a
        flow that will not work. On the changelist Django asks per selected
        object and refuses the *whole* bulk action if any one of them is
        refused — all-or-nothing, and its call rather than ours. That is why the
        Experiments column exists: an administrator whose selection was refused
        needs to see which one without opening each account.
        """
        if obj is not None and obj.experiments.exists():
            return False
        return super().has_delete_permission(request, obj)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        """Say why there is no delete button, on the page where it is missing.

        Without this the guard is silent: the button is simply absent, which
        reads as a bug or a missing permission rather than a deliberate refusal
        with something to do instead.
        """
        user = self.get_object(request, object_id)
        if user is not None and user.experiments.exists():
            self.message_user(request, _KEPT % {"who": user.get_username()},
                              level="WARNING")
        return super().change_view(request, object_id, form_url, extra_context)


#: Why the account is still there, and what to do instead. Deactivating is the
#: ordinary way somebody leaves: it blocks signing in, and leaves every
#: experiment owned by the person who made it.
_KEPT = (
    "%(who)s still owns experiments, so this account cannot be deleted — "
    "deleting it would leave that work in no group, where nobody can reach it. "
    "Clear the “Active” box below to stop them signing in while their "
    "experiments stay theirs, or reassign the experiments to somebody else "
    "first."
)
