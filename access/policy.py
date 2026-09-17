"""Who sees whose work.

The default `EXPERIMENT_POLICY`. Like the login wall, it is installed
unconditionally and inert until `REQUIRE_LOGIN` is on — so an install with no
accounts stays exactly as open as `OpenPolicy` leaves it, and turning the switch
on remains one variable rather than a checklist.

An instance hosted for several research groups exists so that the groups do not
see each other, so almost everything here narrows. An experiment's group is its
**owner's** group; there is no second place that fact is recorded, and nothing
moves an experiment between groups except changing who owns it.

What a signed-in person can reach:

* **Their own.** Anything, always.
* **Shared with their group**, by somebody in the same group (`shared`).
* **Shared with them by name** (`shared_with`), whoever that person is — this
  one deliberately crosses the group boundary, because a named invitation is an
  act of judgement by the owner rather than a property of a group.
* **Everything in their group**, if they are its lead. Which the lead sees
  *where* is the view's business, not this one's — see `for_listing`.
* **Nothing at all**, if they are a site admin: that role manages groups, and
  showing it colleagues' experiments is the thing the separation is for.

Two things that are not rules here and are worth saying so:

`owner IS NULL` used to mean "everyone's", which was right when the only
ownerless experiments were the ones predating accounts. With groups it is a leak
by construction, so it means *nobody's* now, and the migration that introduced
groups gave the existing ones an owner rather than leaving them to this.

`view_all_experiments` and `manage_experiments` still cut across every group.
They are an escape hatch, granted to nobody; see `access/models.py`.
"""

from django.conf import settings
from django.db.models import Q
from django.utils.translation import gettext_lazy as _

from ui.models import Experiment
from ui.permissions import EXPORT, OpenPolicy, VIEW

#: Manages groups, their size and their usage. Sees no experiments.
MANAGE_SITE = "access.manage_site"
#: May change the settings every inheriting experiment on the instance follows.
CHANGE_DEFAULTS = "access.change_defaults"
#: May bring custom-model code — arbitrary code execution — onto the instance.
USE_CUSTOM_MODELS = "access.use_custom_models"
#: The escape hatch, across every group. Granted to nobody by default.
VIEW_ALL = "access.view_all_experiments"
MANAGE_ALL = "access.manage_experiments"


def membership_of(user):
    """*user*'s membership, or None if they are in no group.

    None is an ordinary answer, not an error: a site admin has no group by
    design, and a superuser created with stock `createsuperuser` has none yet.
    """
    if user is None or not getattr(user, "is_authenticated", False):
        return None
    return getattr(user, "membership", None)


def is_site_admin(user) -> bool:
    return _holds(user, MANAGE_SITE)


def is_lead(user) -> bool:
    membership = membership_of(user)
    return bool(membership and membership.is_lead)


class GroupPolicy(OpenPolicy):
    """Experiments belong to their owner, and are visible within a group."""

    # ── which experiments exist, for this request ────────────────────────────

    def experiments(self, request):
        """Everything this request may *reach*.

        Authorization, not presentation: a group lead reaches their colleagues'
        experiments through this, and `for_listing` is what keeps them off the
        lead's own Experiments page. Keeping the two apart is what lets there be
        one rule about what may be touched and several pages that show different
        slices of it.
        """
        if not settings.REQUIRE_LOGIN:
            return Experiment.objects.all()

        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            # The wall should have caught this. Returning nothing rather than
            # trusting that it did keeps one missing exemption from becoming a
            # data leak.
            return Experiment.objects.none()

        if _holds(user, VIEW_ALL) or _holds(user, MANAGE_ALL):
            return Experiment.objects.all()

        # Checked before the membership, because a site admin who also happened
        # to have one would otherwise see that group's work — and the role is
        # defined by not seeing anyone's.
        if is_site_admin(user):
            return Experiment.objects.none()

        membership = membership_of(user)
        if membership is None:
            # In no group and not a site admin: their own, and whatever has
            # been shared with them by name. Not "everything unowned", which is
            # what this used to fall through to.
            return Experiment.objects.filter(
                Q(owner=user) | Q(shared_with=user)).distinct()

        if membership.is_lead:
            reach = Q(owner__membership__group=membership.group)
        else:
            reach = (Q(owner=user)
                     | Q(shared=True, owner__membership__group=membership.group))
        # By name, from anyone at all: an invitation the owner made deliberately
        # is not bounded by the group, or it would not be an invitation.
        return Experiment.objects.filter(reach | Q(shared_with=user)).distinct()

    def for_listing(self, request):
        """The experiments a page should *list* — the sidebar, the index.

        The same as `experiments()` for everyone except a group lead, whose
        everyday view stays their own work. Their colleagues' is a panel of its
        own, so that being a lead does not quietly turn the main list into
        everyone's.
        """
        visible = self.experiments(request)
        if not settings.REQUIRE_LOGIN or not is_lead(getattr(request, "user", None)):
            return visible
        user = request.user
        return visible.filter(Q(owner=user) | Q(shared=True) | Q(shared_with=user))

    def group_experiments(self, request):
        """A lead's colleagues' experiments — the other half of the split."""
        membership = membership_of(getattr(request, "user", None))
        if membership is None or not membership.is_lead:
            return Experiment.objects.none()
        return Experiment.objects.filter(
            owner__membership__group=membership.group).exclude(owner=request.user)

    # ── and what may be done to one ──────────────────────────────────────────

    def may(self, request, experiment, action):
        if not settings.REQUIRE_LOGIN:
            return True

        user = request.user
        if _holds(user, MANAGE_ALL):
            return True

        # Anything the queryset returned may be looked at, and taken away: an
        # export carries only what its page already shows (server paths are
        # stripped on the way out).
        if action in (VIEW, EXPORT):
            return True

        if experiment.owner_id == user.pk:
            return True

        # A lead may act on anything in their group — which is the point of the
        # role: stopping a run that is going wrong should not need its owner.
        membership = membership_of(user)
        if membership and membership.is_lead:
            owner_membership = membership_of(experiment.owner)
            return bool(owner_membership
                        and owner_membership.group_id == membership.group_id)
        return False

    # ── questions that are not about one experiment ──────────────────────────

    def may_upload_models(self, request):
        if not settings.ALLOW_CUSTOM_MODELS:
            return False
        if not settings.REQUIRE_LOGIN:
            return True
        return _holds(getattr(request, "user", None), USE_CUSTOM_MODELS)

    def custom_model_refusal(self, experiment, user):
        """The check that actually matters, because it is also made in the
        worker where there is no form to have been rendered.

        `ALLOW_CUSTOM_MODELS` being off is not handled here — that is an
        instance-wide mechanic the run engine already refuses on. This is only
        the question of *whose* code is about to be executed.
        """
        if not settings.REQUIRE_LOGIN or not experiment.model_file:
            return ""
        if _holds(user, USE_CUSTOM_MODELS):
            return ""
        who = user.get_username() if getattr(user, "is_authenticated", False) else None
        if who:
            return _("%(user)s is not allowed to run custom models here.") % {"user": who}
        return _("This experiment's custom model has no account behind it, so it "
                 "cannot be run. Give the experiment an owner who is allowed to "
                 "run custom models.")

    def may_change_defaults(self, request):
        """These are the defaults every inheriting experiment on the instance
        follows, so one person changing them changes everyone's pages."""
        if not settings.REQUIRE_LOGIN:
            return True
        return _holds(getattr(request, "user", None), CHANGE_DEFAULTS)


#: The name it had before groups, kept so an `EXPERIMENT_POLICY` written against
#: the old one keeps resolving. The rules are the group rules now; there is no
#: version of this that is only about ownership any more.
OwnerPolicy = GroupPolicy


def _holds(user, permission: str) -> bool:
    """Whether *user* is signed in and holds *permission*.

    The authentication check is not redundant with `has_perm`: `AnonymousUser`
    answers it too, and always False — but `user` here can also be None, which
    `has_perm` cannot be asked at all.
    """
    return bool(user and getattr(user, "is_authenticated", False)
                and user.has_perm(permission))
