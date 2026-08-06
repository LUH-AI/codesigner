"""Ownership and sharing.

The default `EXPERIMENT_POLICY`. Like the login wall, it is installed
unconditionally and inert until `REQUIRE_LOGIN` is on — so an install with no
accounts stays exactly as open as `OpenPolicy` leaves it, and turning the switch
on remains one variable rather than a checklist of two.

Three kinds of experiment once accounts exist:

* **Yours.** You can do anything to it.
* **Shared with you.** Visible and exportable; not runnable, editable or
  deletable. Sharing is an invitation to look, not a transfer of control — the
  owner's results should not change because a colleague pressed Run.
* **Nobody's** (`owner IS NULL`). Everyone's, in full. These are the experiments
  that existed before the instance had accounts, so there is no owner whose
  wishes are being overridden, and hiding them would silently swallow an
  operator's existing work the moment they flipped the switch. An operator
  assigns owners in the admin; after that the normal rules apply.

Staff are exempt from the action rules but not from arithmetic: they still only
see what the queryset returns, and it returns everything for them.
"""

from django.conf import settings
from django.db.models import Q

from ui.models import Experiment
from ui.permissions import EXPORT, OpenPolicy, VIEW


class OwnerPolicy(OpenPolicy):
    """Experiments belong to the account that created them."""

    def experiments(self, request):
        if not settings.REQUIRE_LOGIN:
            return Experiment.objects.all()

        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            # The wall should have caught this. Returning nothing rather than
            # trusting that it did keeps one missing exemption from becoming a
            # data leak.
            return Experiment.objects.none()

        if user.is_staff:
            return Experiment.objects.all()
        return Experiment.objects.filter(
            Q(owner=user) | Q(owner__isnull=True) | Q(shared=True))

    def may(self, request, experiment, action):
        if not settings.REQUIRE_LOGIN:
            return True

        # Anything the queryset returned may be looked at, and taken away: an
        # export of a shared experiment carries only what its page already
        # shows (server paths are stripped on the way out).
        if action in (VIEW, EXPORT):
            return True

        user = request.user
        return (experiment.owner_id in (None, user.pk)) or user.is_staff
