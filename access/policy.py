"""`GroupPolicy`, the default `EXPERIMENT_POLICY`, stays inert until
`REQUIRE_LOGIN` is on. A signed-in person reaches their own experiments, grants
(`ExperimentShare`) in a group they are still in, and everything in a group they
lead (`Experiment.group`). A viewer may view and export, a contributor may also
run, edit and delete, and a lead may run and edit. The owner alone holds `SHARE`
and `RENAME`, and membership decides a site admin's reach as for anybody else.
"""

from django.conf import settings
from django.db.models import Q
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

from experiments.models import Experiment, ExperimentShare

#: Every route declares one action, so a policy answers per action and the
#: URLconf shows what each route does.
VIEW = "view"        # render page or fragment
RUN = "run"          # start, cancel or prepare its environment
EDIT = "edit"        # change settings
DELETE = "delete"
EXPORT = "export"    # copy off the instance
SHARE = "share"      # grant access or hand over
RENAME = "rename"

ACTIONS = frozenset({VIEW, RUN, EDIT, DELETE, EXPORT, SHARE, RENAME})


class OpenPolicy:
    """`OpenPolicy` gives everyone every experiment, for installs without
    accounts."""

    def experiments(self, request):
        """`experiments` returns the queryset *request* may see and, under every
        policy, leaves deleted experiments to `bin`.
        """
        return Experiment.objects.filter(deleted_at__isnull=True)

    def bin(self, request):
        """`bin` returns the deleted experiments *request* may restore, which
        without accounts means all of them.
        """
        return Experiment.objects.filter(deleted_at__isnull=False)

    def trash(self, request):
        return Experiment.objects.none()

    def for_listing(self, request):
        """`for_listing` returns what a page should list, which may be narrower
        than what is reachable. `OpenPolicy` lists everything it reaches.
        """
        return self.experiments(request)

    def may(self, request, experiment, action) -> bool:
        """`may` allows every action."""
        return True

    # ── instance-wide questions ──────────────────────────────────────────────

    def may_upload_models(self, request):
        """`ALLOW_CUSTOM_MODELS` is the floor because an uploaded model runs
        arbitrary code; policies with accounts can require more.
        """
        return settings.ALLOW_CUSTOM_MODELS

    def custom_model_refusal(self, experiment, user) -> str:
        """`custom_model_refusal` returns why *user* is barred from
        *experiment*'s custom model, or "", since a worker log needs a reason.
        """
        return ""

    def may_change_defaults(self, request) -> bool:
        return True


_cached: tuple[str, OpenPolicy] | None = None


def policy() -> OpenPolicy:
    """`policy` caches one instance per `EXPERIMENT_POLICY` value, so a test can
    repoint the setting.
    """
    global _cached
    path = getattr(settings, "EXPERIMENT_POLICY", None) or (
        f"{OpenPolicy.__module__}.{OpenPolicy.__qualname__}")
    if _cached is None or _cached[0] != path:
        _cached = (path, import_string(path)())
    return _cached[1]


def visible_experiments(request):
    """`visible_experiments` lists experiments for enumerating pages through
    `for_listing`, which keeps a lead's group work separate.
    """
    return policy().for_listing(request)

#: Manages groups; grants no experiment access.
MANAGE_SITE = "access.manage_site"
#: May change the instance-wide default settings.
CHANGE_DEFAULTS = "access.change_defaults"
#: May upload custom models, which run arbitrary code.
USE_CUSTOM_MODELS = "access.use_custom_models"
#: Escape hatch across groups; off by default.
VIEW_ALL = "access.view_all_experiments"
MANAGE_ALL = "access.manage_experiments"


def memberships_of(user) -> list:
    """`memberships_of` caches *user*'s memberships, groups included, on the
    user, so a page costs one query. An empty list is normal, since a site admin
    may be in no group.
    """
    if user is None or not getattr(user, "is_authenticated", False):
        return []
    cached = getattr(user, "_memberships", None)
    if cached is None:
        cached = list(user.memberships.select_related("group").all())
        user._memberships = cached
    return cached


def group_ids_of(user) -> set:
    return {m.group_id for m in memberships_of(user)}


def led_group_ids(user) -> set:
    """`led_group_ids` returns the ids of the groups *user* leads, for
    `group_id__in` filters.
    """
    return {m.group_id for m in memberships_of(user) if m.is_lead}


def primary_led_group_ids(user) -> set:
    return {m.group_id for m in memberships_of(user) if m.is_primary_lead}


def membership_in(user, group_id):
    for membership in memberships_of(user):
        if membership.group_id == group_id:
            return membership
    return None


def groups_to_choose_from(user) -> list:
    """`groups_to_choose_from` returns *user*'s groups when two or more exist
    and an empty list otherwise, because `Experiment.save` handles a single
    group."""
    if not settings.REQUIRE_LOGIN:
        return []
    groups = [m.group for m in memberships_of(user)]
    return sorted(groups, key=lambda g: g.name.lower()) if len(groups) > 1 else []


def is_site_admin(user) -> bool:
    return _holds(user, MANAGE_SITE)


def is_lead(user) -> bool:
    """`is_lead` asks whether *user* leads any group, for the layout; rules ask
    `is_lead_of`.
    """
    return bool(led_group_ids(user))


def is_lead_of(user, group_id) -> bool:
    return group_id is not None and group_id in led_group_ids(user)


def is_primary_lead(user) -> bool:
    return bool(primary_led_group_ids(user))


class GroupPolicy(OpenPolicy):
    """`GroupPolicy` extends `OpenPolicy` with owners and groups."""

    # ── which experiments exist ──────────────────────────────────────────────

    def experiments(self, request):
        """`experiments` returns what *request* may reach, minus other people's
        drafts, which stay their owner's alone.
        """
        reachable = self._reachable(request)
        user = getattr(request, "user", None)
        if not settings.REQUIRE_LOGIN or user is None or not user.is_authenticated:
            return reachable
        return reachable.exclude(Q(draft=True) & ~Q(owner=user))

    def _reachable(self, request):
        """`_reachable` answers authorization only; `for_listing` decides which
        reachable experiments each page shows.
        """
        if not settings.REQUIRE_LOGIN:
            return super().experiments(request).select_related("data")

        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            # An anonymous user gets nothing, so one missing wall exemption
            # cannot leak data.
            return Experiment.objects.none()

        # The escape hatch also reaches a deleted group's work, for support
        # cases, and excludes what a person deleted.
        if _holds(user, VIEW_ALL) or _holds(user, MANAGE_ALL):
            return Experiment.objects.filter(deleted_at__isnull=True)

        # The queries below see only live work, and a deleted group's
        # experiments wait in `trash()` for a site admin. The `data` join saves
        # one query per row, since every list shows experiment names.
        live = Experiment.objects.select_related("data").filter(
            trashed_at__isnull=True, deleted_at__isnull=True)

        # A person in no group reaches only their own work. Grants and lead
        # reach are checked per group.
        return live.filter(Q(owner=user)
                           | _granted(user)
                           | Q(group_id__in=led_group_ids(user))).distinct()

    def trash(self, request):
        """`trash` returns the site bin for a site admin: a deleted group's
        work, plus deletions no person's bin still holds.
        """
        if not settings.REQUIRE_LOGIN:
            return Experiment.objects.none()
        if not is_site_admin(getattr(request, "user", None)):
            return Experiment.objects.none()
        # The site bin receives a deleted group's work unless a person deleted
        # it first and still holds it in a bin. `orphaned()` adds deletions that
        # no bin holds any more.
        from experiments.services.bin import orphaned
        group_gone = Experiment.objects.filter(trashed_at__isnull=False,
                                               deleted_at__isnull=True)
        return (group_gone | orphaned()).distinct()

    def bin(self, request):
        """`bin` returns what the requesting user deleted or had deleted under
        them, from their `BinEntry` rows.
        """
        if not settings.REQUIRE_LOGIN:
            return super().bin(request)
        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            return Experiment.objects.none()
        from experiments.services.bin import cutoff
        return (Experiment.objects.select_related("data")
                .filter(deleted_at__gte=cutoff(), bin_entries__user=user))

    def for_listing(self, request):
        """`for_listing` matches `experiments()` except for a group lead, whose
        main list keeps to their own and shared work.
        """
        visible = self.experiments(request)
        if not settings.REQUIRE_LOGIN or not is_lead(getattr(request, "user", None)):
            return visible
        user = request.user
        return visible.filter(Q(owner=user) | _granted(user))

    def group_experiments(self, request, group_id=None):
        """`group_experiments` returns a lead's colleagues' experiments in one
        led group, or in all of them when *group_id* is None. A *group_id*
        outside the user's led groups yields an empty queryset.
        """
        user = getattr(request, "user", None)
        led = led_group_ids(user)
        ids = led if group_id is None else led & {group_id}
        if not ids:
            return Experiment.objects.none()
        return (Experiment.objects.select_related("data")
                .filter(group_id__in=ids, trashed_at__isnull=True,
                        deleted_at__isnull=True, draft=False)
                .exclude(owner=user))

    # ── per-experiment checks ────────────────────────────────────────────────

    def may(self, request, experiment, action):
        if not settings.REQUIRE_LOGIN:
            return True

        user = request.user
        if _holds(user, MANAGE_ALL):
            return True

        # Every experiment in the queryset may be viewed and exported, since an
        # export strips server paths.
        if action in (VIEW, EXPORT):
            return True

        if experiment.owner_id == user.pk:
            return True
        if action in (SHARE, RENAME):
            return False

        contributor = share_of(user, experiment) == ExperimentShare.CONTRIBUTOR
        if action == DELETE:
            # A lead deletes only as a contributor, since deletion belongs to
            # the work's people.
            return contributor
        # Lead reach applies in *its* group only.
        return contributor or (action in (RUN, EDIT)
                               and is_lead_of(user, experiment.group_id))

    # ── instance-wide questions ──────────────────────────────────────────────

    def may_upload_models(self, request):
        if not settings.ALLOW_CUSTOM_MODELS:
            return False
        if not settings.REQUIRE_LOGIN:
            return True
        return _holds(getattr(request, "user", None), USE_CUSTOM_MODELS)

    def custom_model_refusal(self, experiment, user):
        """`custom_model_refusal` runs in the worker too, where no form exists,
        and only asks whose code runs; the run engine handles
        `ALLOW_CUSTOM_MODELS`.
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
        """`may_change_defaults` guards the defaults every inheriting experiment
        shares."""
        if not settings.REQUIRE_LOGIN:
            return True
        return _holds(getattr(request, "user", None), CHANGE_DEFAULTS)


#: `OwnerPolicy` is the pre-group name, kept so old `EXPERIMENT_POLICY` values
#: still resolve.
OwnerPolicy = GroupPolicy


def share_of(user, experiment):
    """`share_of` returns *user*'s grant level on *experiment*, or None after
    *user* leaves the group. The check also covers memberships removed without
    the signal.
    """
    if experiment.group_id not in group_ids_of(user):
        return None
    return (experiment.shares.filter(user=user)
            .values_list("level", flat=True).first())


def _granted(user):
    """`_granted` uses a subquery because a join through `shares` would repeat
    an experiment once per grant.
    """
    return Q(pk__in=ExperimentShare.objects.filter(user=user).values("experiment_id"),
             group_id__in=group_ids_of(user))


def _holds(user, permission: str) -> bool:
    """`_holds` checks authentication first because *user* may be None, which
    has no `has_perm`.
    """
    return bool(user and getattr(user, "is_authenticated", False)
                and user.has_perm(permission))
