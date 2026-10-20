from django.conf import settings
from django.contrib import messages
from django.http import Http404, HttpResponseBadRequest
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from access.policy import SHARE
from experiments.models import ExperimentShare
from experiments.services import sharing

from ..permissions import experiment_view


def _back_to_sharing(request, exp):
    """Wherever the sharing controls were — the group page, as a rule — or the
    experiment itself when the form did not say."""
    from ..navigation import safe_next
    return redirect(safe_next(request) or reverse("ui:experiment_detail", args=[exp.pk]))


def _colleague(request, exp):
    """The account a sharing form names, from the experiment's own group.

    Looked up among the group's members and nowhere else, so an id from outside
    the group is simply not found — the same answer as one that does not exist.
    """
    from django.contrib.auth import get_user_model
    if not settings.REQUIRE_LOGIN or exp.group_id is None:
        raise Http404
    try:
        pk = int(request.POST.get("user", ""))
    except ValueError:
        raise Http404
    colleague = (get_user_model().objects
                 .filter(pk=pk, memberships__group_id=exp.group_id).first())
    if colleague is None or colleague.pk == exp.owner_id:
        raise Http404
    return colleague


@require_POST
@experiment_view(SHARE)
def experiment_share(request, exp):
    """Set one colleague's access to this experiment: none, viewer or contributor.

    One person at a time, and only somebody in the experiment's group. The
    owner's alone to decide — a contributor can do everything to the work except
    choose who else may.
    """
    colleague = _colleague(request, exp)
    level = request.POST.get("level", "")
    if level not in dict(ExperimentShare.LEVELS) and level != "none":
        return HttpResponseBadRequest(_("Unknown sharing level."))
    sharing.set_access(exp, colleague, level, granted_by=request.user)
    return _back_to_sharing(request, exp)


@require_POST
@experiment_view(SHARE)
def experiment_transfer(request, exp):
    """Hand this experiment to one of its contributors.

    Only a contributor: somebody who can already run and delete it, so the one
    new power ownership gives them is deciding who else may. The previous owner
    is left a contributor, not shut out — handing over the work is not the same
    decision as walking away from it — and both happen together or not at all,
    so there is never a moment with no owner or with two.
    """
    colleague = _colleague(request, exp)
    # The page offers this for contributors only, so anybody else is a request
    # it never made — not found, the same answer as a stranger.
    if not sharing.transfer(exp, colleague):
        raise Http404
    messages.success(request, _("%(name)s now belongs to %(user)s. You are a "
                                "contributor.")
                     % {"name": exp.title, "user": colleague.get_username()})
    return _back_to_sharing(request, exp)
