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
* **Shared with them** (`ExperimentShare`), by its owner, at a level: a viewer
  may look and export, a contributor may also run, change and delete it. Only
  within the experiment's group, and only while they are still in it.
* **Everything in a group they lead.** They may look at it, run and stop it —
  stopping a run that is going wrong should not need its owner — but not delete
  it: that stays with the people whose work it is. Which the lead sees *where*
  is the view's business, not this one's — see `for_listing`.

Deciding who else may reach an experiment, and handing it to somebody, is the
owner's alone (`SHARE`). A contributor can do everything to the work but choose
who else does.

A site admin is not a fourth case. Managing groups and being in one are separate
facts, so the role is read from `manage_site` and the reach from the membership,
exactly as for anybody else: a site admin with a group of their own sees that
group and no other, and one with no membership sees nothing because there is
nothing of theirs to see. This used to be a rule — site admins were shown
nothing whatever they belonged to — and it made the ordinary case, an operator
who also runs experiments, impossible to express.

Two things that are not rules here and are worth saying so:

`owner IS NULL` used to mean "everyone's", which was right when the only
ownerless experiments were the ones predating accounts. With groups it is a leak
by construction, so it means *nobody's* now, and the migration that introduced
groups gave the existing ones an owner rather than leaving them to this.

An experiment's group is `Experiment.group`, not its owner's membership. The two
agreed until an owner was deleted or moved group; see `ui/migrations/0024`.

`view_all_experiments` and `manage_experiments` still cut across every group.
They are an escape hatch, granted to nobody; see `access/models.py`.
"""

from django.conf import settings
from django.db.models import Q
from django.utils.translation import gettext_lazy as _

from ui.models import Experiment, ExperimentShare
from ui.permissions import DELETE, EDIT, EXPORT, OpenPolicy, RUN, SHARE, VIEW

#: Manages groups, their size and their usage. Sees no experiments.
MANAGE_SITE = "access.manage_site"
#: May change the settings every inheriting experiment on the instance follows.
CHANGE_DEFAULTS = "access.change_defaults"
#: May bring custom-model code — arbitrary code execution — onto the instance.
USE_CUSTOM_MODELS = "access.use_custom_models"
#: The escape hatch, across every group. Granted to nobody by default.
VIEW_ALL = "access.view_all_experiments"
MANAGE_ALL = "access.manage_experiments"


def memberships_of(user) -> list:
    """Every membership *user* holds, each with its group, cached on the user.

    A list rather than a queryset: every caller wants all of them, and a
    request's `user` is one object for the life of the request, so caching here
    makes a page one query rather than one per question it asks.

    Empty is an ordinary answer, not an error — a site admin need not be in a
    group, and a stock `createsuperuser` is in none yet.
    """
    if user is None or not getattr(user, "is_authenticated", False):
        return []
    cached = getattr(user, "_memberships", None)
    if cached is None:
        cached = list(user.memberships.select_related("group").all())
        user._memberships = cached
    return cached


def group_ids_of(user) -> set:
    """The groups *user* belongs to, as ids — the unit every `__in` wants."""
    return {m.group_id for m in memberships_of(user)}


def led_group_ids(user) -> set:
    """The groups *user* leads. The unit of a lead's reach, everywhere.

    Ids rather than groups, because every consumer is a `group_id__in=` —
    names are a panel's business, not a rule's.
    """
    return {m.group_id for m in memberships_of(user) if m.is_lead}


def primary_led_group_ids(user) -> set:
    return {m.group_id for m in memberships_of(user) if m.is_primary_lead}


def membership_in(user, group_id):
    """*user*'s membership in one named group, or None.

    What `membership_of` used to answer, now that the question has to say which
    group it is about. Somebody in two groups has no "their membership", and
    picking one would be deciding for them which boundary applies.
    """
    for membership in memberships_of(user):
        if membership.group_id == group_id:
            return membership
    return None


def groups_to_choose_from(user) -> list:
    """The groups a new experiment of *user*'s could go in, when that is a
    question — two or more. With one, `Experiment.save` files it there; with
    none, or no accounts, there is nothing to choose."""
    if not settings.REQUIRE_LOGIN:
        return []
    groups = [m.group for m in memberships_of(user)]
    return sorted(groups, key=lambda g: g.name.lower()) if len(groups) > 1 else []


def is_site_admin(user) -> bool:
    return _holds(user, MANAGE_SITE)


def is_lead(user) -> bool:
    """Leads *any* group. The question the layout asks, not the rules.

    Whether a tab exists is about whether somebody leads something. Whether
    they may act on a given experiment is about whether they lead *its* group,
    and that is `is_lead_of` — the two used to be one function because a
    person could only lead one thing.
    """
    return bool(led_group_ids(user))


def is_lead_of(user, group_id) -> bool:
    """Leads *this* group. Every authorization question asks this one."""
    return group_id is not None and group_id in led_group_ids(user)


def is_primary_lead(user) -> bool:
    """Primary lead of any group."""
    return bool(primary_led_group_ids(user))


def is_primary_lead_of(user, group_id) -> bool:
    """The one lead of *this* group who decides who else leads it.

    Every other question about reach asks `is_lead_of`, which this implies.
    Only the views that appoint and unmake leads ask for this.
    """
    return group_id is not None and group_id in primary_led_group_ids(user)


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
            return super().experiments(request).select_related("data")

        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            # The wall should have caught this. Returning nothing rather than
            # trusting that it did keeps one missing exemption from becoming a
            # data leak.
            return Experiment.objects.none()

        # The escape hatch, and the only thing that reaches a deleted group's
        # work: a support case is exactly when somebody needs to see what was
        # set aside. Not what a person deleted, which is theirs to settle.
        if _holds(user, VIEW_ALL) or _holds(user, MANAGE_ALL):
            return Experiment.objects.filter(deleted_at__isnull=True)

        # Everything below is about live work. An experiment whose group was
        # deleted is nobody's to reach until a site admin rehomes it, which is
        # the one thing the bin is for; `trash()` is where it is looked at.
        # `data` on every list: reading an experiment's name is not optional
        # in practice — the sidebar, the index and every breadcrumb do it —
        # so without the join that is one extra query per row.
        live = Experiment.objects.select_related("data").filter(
            trashed_at__isnull=True, deleted_at__isnull=True)

        # Three clauses. A person in no group reaches only their own — not
        # "everything unowned", which is what this used to fall through to. A
        # site admin in no group sees nothing, which is right: there is nothing
        # of theirs.
        #
        # The other two are per group, and a person may be in several: a grant
        # in a group they are in, everything in a group they lead. Asked of
        # each group separately, so a lead of one group and a member of
        # another reaches exactly what each role grants in each.
        return live.filter(Q(owner=user)
                           | _granted(user)
                           | Q(group_id__in=led_group_ids(user))).distinct()

    def trash(self, request):
        """The site's bin: what a group's deletion left behind, and what
        somebody deleted that nobody kept, for a site admin to settle.

        Not part of `experiments()` and deliberately not reachable from it: a
        trashed experiment is out of every group, so every rule above would have
        to special-case it. It is its own question, asked by its own panel, and
        answered for one role.
        """
        if not settings.REQUIRE_LOGIN:
            return Experiment.objects.none()
        if not is_site_admin(getattr(request, "user", None)):
            return Experiment.objects.none()
        # Two ways in. A deleted group's work, unless somebody had deleted it
        # first — that is still in their bins, and comes here when one of them
        # restores it. And what somebody deleted that nobody's bin holds any
        # more: every one of its people took it out, or it waited too long.
        from ui.services.bin import orphaned
        group_gone = Experiment.objects.filter(trashed_at__isnull=False,
                                               deleted_at__isnull=True)
        return (group_gone | orphaned()).distinct()

    def bin(self, request):
        """What this person deleted, or had deleted from under them.

        One `BinEntry` per person, written when it was deleted — see
        `ui/models.py` on why it is recorded rather than worked out again.
        """
        if not settings.REQUIRE_LOGIN:
            return super().bin(request)
        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            return Experiment.objects.none()
        # Only for as long as a bin keeps it — see `ui/services/bin.py`.
        from ui.services.bin import cutoff
        return (Experiment.objects.select_related("data")
                .filter(deleted_at__gte=cutoff(), bin_entries__user=user))

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
        return visible.filter(Q(owner=user) | _granted(user))

    def group_experiments(self, request, group_id=None):
        """A lead's colleagues' experiments — the other half of the split.

        In one named group, or across every group they lead when none is named.
        Narrowed to groups they actually lead either way: naming a group is a
        question about it, not a way into it.
        """
        user = getattr(request, "user", None)
        led = led_group_ids(user)
        ids = led if group_id is None else led & {group_id}
        if not ids:
            return Experiment.objects.none()
        return (Experiment.objects.select_related("data")
                .filter(group_id__in=ids, trashed_at__isnull=True,
                        deleted_at__isnull=True)
                .exclude(owner=user))

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
        if action == SHARE:
            return False

        contributor = share_of(user, experiment) == ExperimentShare.CONTRIBUTOR
        if action == DELETE:
            # Not a lead, unless they are also one of these: deleting is a
            # decision about the work, and the work is its people's.
            return contributor
        # *Its* group, not theirs: somebody who leads one group and is a member
        # of another is a lead only where they lead.
        return contributor or (action in (RUN, EDIT)
                               and is_lead_of(user, experiment.group_id))

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
        if not settings.REQUIRE_LOGIN or not experiment.data.model_file:
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


def share_of(user, experiment):
    """The level *user* was granted on *experiment*, or None.

    None as well for a grant whose holder has since left the experiment's
    group. Leaving deletes the grants in it (`access/signals.py`); asking again
    here keeps a membership removed some other way from leaving access behind.
    """
    if experiment.group_id not in group_ids_of(user):
        return None
    return (experiment.shares.filter(user=user)
            .values_list("level", flat=True).first())


def _granted(user):
    """Shared with *user*, in a group they are still in — see `share_of`.

    A subquery rather than a join through `shares`: joining yields one row per
    grant on the experiment, so any queryset built on this without its own
    `distinct()` lists a shared experiment once per person it is shared with.
    """
    return Q(pk__in=ExperimentShare.objects.filter(user=user).values("experiment_id"),
             group_id__in=group_ids_of(user))


def _holds(user, permission: str) -> bool:
    """Whether *user* is signed in and holds *permission*.

    The authentication check is not redundant with `has_perm`: `AnonymousUser`
    answers it too, and always False — but `user` here can also be None, which
    `has_perm` cannot be asked at all.
    """
    return bool(user and getattr(user, "is_authenticated", False)
                and user.has_perm(permission))
