"""`@experiment_view` is the one route from a view to an experiment, and
`tests/ui/test_permissions.py` audits the URLconf for it. An experiment outside
the policy's queryset gets a 404, since a 403 would reveal that the experiment
exists. A forbidden action on a visible experiment gets a plain 403.
"""

from functools import wraps
from typing import Any, Callable

from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect

from access.policy import ACTIONS, policy

#: `CREATED`, the default, limits a route to created experiments, since drafts
#: are worked on only in their setup.
CREATED = "created"
#: Setup routes: drafts only.
ONLY = "only"
#: Both: the prior figure's requests.
BOTH = "both"
DRAFTS = (CREATED, ONLY, BOTH)


class WrongKind(Http404):
    """`WrongKind` is the 404 for a draft reached where a created experiment
    was asked for, or the reverse. A page catches it and sends the reader to
    where that kind is worked on.
    """

    def __init__(self, experiment):
        super().__init__("not reached by this route")
        self.experiment = experiment


def resolve_experiment(request, pk, action, drafts=CREATED):
    """`resolve_experiment` runs the check of `experiment_view` for code that
    reaches an experiment outside a view. A pk the policy hides raises
    `Http404`, the wrong kind raises `WrongKind`, and a refused action raises
    `PermissionDenied`.
    """
    current = policy()
    exp = get_object_or_404(current.experiments(request), pk=pk)
    if (exp.draft and drafts == CREATED) or (not exp.draft and drafts == ONLY):
        raise WrongKind(exp)
    if not current.may(request, exp, action):
        raise PermissionDenied
    return exp


def experiment_view(action, drafts=CREATED):
    """`experiment_view` resolves `<int:pk>` to an experiment the request may
    act on, and calls the view with that experiment in place of `pk`. Therefore
    a view only receives a checked experiment. A GET for the wrong kind (see
    `DRAFTS`) redirects a draft to its setup and a created experiment to its
    dashboard. Other methods get a 404.
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    if drafts not in DRAFTS:
        raise ValueError(f"unknown drafts {drafts!r}; expected one of {DRAFTS}")

    def decorate(view: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view)
        def wrapped(request, pk, *args, **kwargs):
            try:
                exp = resolve_experiment(request, pk, action, drafts)
            except WrongKind as wrong:
                if request.method != "GET":
                    raise
                return redirect("ui:setup_resume" if wrong.experiment.draft
                                else "ui:experiment_detail", pk=wrong.experiment.pk)
            return view(request, exp, *args, **kwargs)

        #: The URLconf audit reads these; `wraps` keeps them past
        #: `require_POST`.
        setattr(wrapped, "experiment_action", action)
        setattr(wrapped, "experiment_drafts", drafts)
        return wrapped

    return decorate
