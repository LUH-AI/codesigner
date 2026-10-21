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
from typing import Any, Callable

from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect

from access.policy import ACTIONS, policy

#: Which experiments a route reaches, by whether they are drafts (still being
#: set up — see ui/setup_steps.py). Every route reaches created experiments
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

    def decorate(view: Callable[..., Any]) -> Callable[..., Any]:
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
        setattr(wrapped, "experiment_action", action)
        setattr(wrapped, "experiment_drafts", drafts)
        return wrapped

    return decorate
