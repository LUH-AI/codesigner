"""Who may see and do what to an experiment.

Codesigner runs locally and on a private network at least as often as it is
hosted, and in those cases there are no users, no owners and nothing to
authorize. So the answer to every question here is "yes" by default, and the
code that asks the questions cannot tell the difference.

The point of this module is that the questions get asked *at all*. There is
exactly one way for a view to reach an experiment — `@experiment_view` — and it
routes through a policy object. Adding real rules later means writing a second
policy and pointing `EXPERIMENT_POLICY` at it, not revisiting a dozen views and
hoping none was missed. The URLconf audit in `tests/ui/test_permissions.py`
makes the "exactly one way" part enforceable rather than aspirational.

Two distinct questions, because they fail differently:

* **Which experiments exist, for this request?** A policy narrows the queryset,
  and a miss is a **404** — an experiment you may not see must be
  indistinguishable from one that is not there, or the URL space itself reports
  who has what.
* **May this request do X to this one?** Asked only about an experiment already
  visible, so its existence is not a secret and a refusal is a plain **403**.
"""

from functools import wraps

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect
from django.utils.module_loading import import_string

from .models import Experiment

#: What a view does with the experiment it is given. Every route declares one,
#: so a policy can answer per action rather than all-or-nothing, and so reading
#: the URLconf tells you what each route is for.
VIEW = "view"        # render it, or a fragment of it
RUN = "run"          # start, cancel, or prepare the environment for a run
EDIT = "edit"        # change its settings
DELETE = "delete"    # destroy it
EXPORT = "export"    # take a copy of it off the instance
SHARE = "share"      # decide who else may reach it, or hand it to somebody
RENAME = "rename"    # change what it is called

ACTIONS = frozenset({VIEW, RUN, EDIT, DELETE, EXPORT, SHARE, RENAME})


class OpenPolicy:
    """Every experiment belongs to everyone. The default, and what a local or
    private-network install wants: no accounts, no owners, no refusals."""

    def experiments(self, request):
        """The experiments this request may see, as a queryset.

        Never a deleted one, under any policy: deleting sets an experiment aside
        into its people's bins, and from then on `bin` is the only way to it.
        """
        return Experiment.objects.filter(deleted_at__isnull=True)

    def bin(self, request):
        """The deleted experiments this request may restore, as a queryset.

        Every one of them, with no accounts — a bin is somebody's, and without
        accounts there is one somebody.
        """
        return Experiment.objects.filter(deleted_at__isnull=False)

    def for_listing(self, request):
        """The experiments a page should *list*, which need not be all of them.

        Authorization and presentation are different questions: something may be
        reachable without belonging on the front page. A policy that draws no
        distinction — this one — answers both the same way, and a policy that
        does (a group lead's own work, versus their whole group's) overrides
        this without touching what may be *reached*.
        """
        return self.experiments(request)

    def may(self, request, experiment, action):
        """Whether this request may perform *action* on a visible experiment."""
        return True

    # ── Questions that are not about one experiment ──────────────────────────

    def may_upload_models(self, request):
        """Whether this request may bring a custom model onto the instance.

        Uploading one is arbitrary code execution, so the instance-wide flag is
        the floor; a policy with accounts can require more on top of it.
        """
        return settings.ALLOW_CUSTOM_MODELS

    def custom_model_refusal(self, experiment, user):
        """Why *user* may not run *experiment*'s custom model, or "" if they may.

        A string rather than a boolean because this is the last thing standing
        between an upload and the interpreter, and "no" without a reason is
        unactionable in a worker log.
        """
        return ""

    def may_change_defaults(self, request):
        """Whether this request may edit the default experiment settings, which
        every inheriting experiment on the instance follows."""
        return True


_cached: tuple[str, object] | None = None


def policy():
    """The configured policy, instantiated once per configured path.

    Keyed on the setting rather than cached outright so a test (or a later
    `access/` install) can point `EXPERIMENT_POLICY` somewhere else and be
    obeyed immediately.
    """
    global _cached
    path = getattr(settings, "EXPERIMENT_POLICY", None) or (
        f"{OpenPolicy.__module__}.{OpenPolicy.__qualname__}")
    if _cached is None or _cached[0] != path:
        _cached = (path, import_string(path)())
    return _cached[1]


def visible_experiments(request):
    """The experiments to list for this request — for the sidebar and anything
    else that enumerates rather than looking one up.

    `for_listing`, not `experiments`: enumerating is the presentation question,
    and the two part company for a group lead, whose everyday list stays their
    own work while their group's is a panel of its own.
    """
    return policy().for_listing(request)


#: Which experiments a route reaches, by whether they are drafts (still being
#: set up — see ui/services/setup.py). Every route reaches created experiments
#: only, unless it says otherwise: a draft has no dashboard to show, no run to
#: start and nothing to export, and its setup is the one place to work on it.
CREATED = "created"
#: Drafts only: the setup's own routes.
ONLY = "only"
#: Both: what the setup's steps share with a created experiment's pages — the
#: prior figure's requests.
BOTH = "both"
DRAFTS = (CREATED, ONLY, BOTH)


def experiment_view(action, drafts=CREATED):
    """Resolve `<int:pk>` to an experiment this request is allowed to act on.

    The decorated view is called with the experiment in place of `pk`:

        @experiment_view(VIEW)
        def experiment_detail(request, exp):
            ...

    which is also why this is worth a decorator rather than a helper call —
    a view cannot accidentally hold a `pk` that was never checked, because it
    is never given one.

    *drafts* says whether the route reaches drafts (see `DRAFTS`). A page
    asked of the wrong kind sends the reader where that kind is worked on — a
    draft to its setup, a created experiment to its dashboard; anything else
    is a 404, as for an experiment that does not exist.
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    if drafts not in DRAFTS:
        raise ValueError(f"unknown drafts {drafts!r}; expected one of {DRAFTS}")

    def decorate(view):
        @wraps(view)
        def wrapped(request, pk, *args, **kwargs):
            current = policy()
            exp = get_object_or_404(current.experiments(request), pk=pk)
            wrong = ((exp.draft and drafts == CREATED)
                     or (not exp.draft and drafts == ONLY))
            if wrong:
                if request.method != "GET":
                    raise Http404
                return redirect("ui:setup_resume" if exp.draft else "ui:experiment_detail",
                                pk=exp.pk)
            if not current.may(request, exp, action):
                raise PermissionDenied
            return view(request, exp, *args, **kwargs)

        #: Read by the URLconf audit. `functools.wraps` copies `__dict__`, so
        #: these survive an outer `require_POST`.
        wrapped.experiment_action = action
        wrapped.experiment_drafts = drafts
        return wrapped

    return decorate
